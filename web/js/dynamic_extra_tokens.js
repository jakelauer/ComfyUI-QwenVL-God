import { app } from "/scripts/app.js";

const NODE_NAME = "AILab_QwenVL_Extra_Tokens";
const MAX_TOKEN_SLOTS = 16;
const HIDDEN_WIDGET_TYPE = "hidden-qwen-token-key";

function valueIndex(name) {
    const match = /^value(\d+)$/.exec(name || "");
    return match ? parseInt(match[1], 10) : null;
}

function keyName(index) {
    return `key${index}`;
}

function valueName(index) {
    return `value${index}`;
}

function findInput(node, name) {
    return node.inputs?.find((input) => input && input.name === name);
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget && widget.name === name);
}

function setWidgetHidden(widget, hidden) {
    if (!widget) {
        return;
    }
    if (hidden) {
        if (widget.type !== HIDDEN_WIDGET_TYPE) {
            widget._qwenOrigType = widget.type;
        }
        widget.type = HIDDEN_WIDGET_TYPE;
        widget.hidden = true;
        widget.computeSize = () => [0, -4];
        return;
    }
    widget.type = widget._qwenOrigType || "text";
    widget.hidden = false;
    if (widget.computeSize) {
        delete widget.computeSize;
    }
}

function decorateKeyWidget(widget, index) {
    if (!widget) {
        return;
    }
    widget.label = `key ${index}`;
    if (widget.options) {
        widget.options.placeholder = `name for value${index}`;
    }
}

function adaptValueType(node, input, connected, linkInfo) {
    if (!input || valueIndex(input.name) == null) {
        return;
    }
    if (connected && linkInfo && node.graph) {
        const origin = node.graph.getNodeById(linkInfo.origin_id);
        const originOut = origin?.outputs?.[linkInfo.origin_slot];
        input.type = originOut?.type && originOut.type !== "*" ? originOut.type : "*";
        return;
    }
    if (!connected) {
        input.type = "*";
    }
}

function ensureValueInput(node, index) {
    const name = valueName(index);
    let input = findInput(node, name);
    if (input) {
        return input;
    }
    node.addInput(name, "*");
    return node.inputs.pop();
}

function reorderInputs(node, wantMax) {
    const values = [];
    const extra = [];
    const rest = [];
    for (const input of node.inputs || []) {
        if (!input) {
            continue;
        }
        const index = valueIndex(input.name);
        if (index != null) {
            if (index >= 1 && index <= wantMax) {
                values[index] = input;
            }
            continue;
        }
        if (input.name === "extra_tokens") {
            extra.push(input);
            continue;
        }
        rest.push(input);
    }

    for (let n = 1; n <= wantMax; n++) {
        if (!values[n]) {
            values[n] = ensureValueInput(node, n);
        }
        values[n].label = valueName(n);
        values[n].hidden = false;
    }

    node.inputs = [
        ...values.slice(1, wantMax + 1),
        ...rest,
        ...extra,
    ];
}

function syncExtraTokenSlots(node) {
    if (!node || !node.inputs || node._qwenTokenSync) {
        return;
    }
    node._qwenTokenSync = true;
    try {
        let highestConnected = 0;
        for (const input of node.inputs) {
            const index = valueIndex(input?.name);
            if (index != null && input.link != null) {
                highestConnected = Math.max(highestConnected, index);
            }
        }
        const wantMax = Math.min(MAX_TOKEN_SLOTS, Math.max(1, highestConnected + 1));
        reorderInputs(node, wantMax);

        for (let n = 1; n <= MAX_TOKEN_SLOTS; n++) {
            const valueInput = findInput(node, valueName(n));
            const connected = !!(valueInput && valueInput.link != null);
            const widget = findWidget(node, keyName(n));
            decorateKeyWidget(widget, n);
            setWidgetHidden(widget, n > wantMax || !connected);
            const keyInput = findInput(node, keyName(n));
            if (keyInput && !keyInput.widget) {
                keyInput.hidden = n > wantMax || !connected;
            }
        }

        if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
            node.setSize(node.computeSize());
        }
        node.graph?.setDirtyCanvas?.(true, true);
    } finally {
        node._qwenTokenSync = false;
    }
}

app.registerExtension({
    name: "QwenVL.dynamicExtraTokens",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            syncExtraTokenSlots(this);
            setTimeout(() => syncExtraTokenSlots(this), 0);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            syncExtraTokenSlots(this);
            setTimeout(() => syncExtraTokenSlots(this), 0);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
            const input = type === 1 ? this.inputs?.[index] : null;
            if (input) {
                adaptValueType(this, input, connected, linkInfo);
            }
            const result = onConnectionsChange?.apply(this, arguments);
            if (type === 1) {
                syncExtraTokenSlots(this);
            }
            return result;
        };
    },
});
