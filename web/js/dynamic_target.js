import { app } from "/scripts/app.js";

const NODE_NAME = "AILab_QwenVL_Prompt_Options";
const INHERIT = "inherit";
const FAMILY_CUSTOM = "✍️ Custom Only (no preset)";
const FAMILY_MINIMAX = "🎬 MiniMax H3";
const FAMILY_WAN = "🎥 Wan 2.2";
const FAMILY_LTX = "🎥 LTX 2.3";
const MODE_I2VA = "I2VA";
const MODE_FL2VA = "FL2VA";
const MODE_R2VA = "R2VA";
const MODE_I2V = "I2V";
const MODE_T2V = "T2V";
const HIDDEN_WIDGET_TYPE = "hidden-qwen-target-mode";

const FAMILIES = [INHERIT, FAMILY_CUSTOM, FAMILY_MINIMAX, FAMILY_WAN, FAMILY_LTX];

const FAMILY_MODES = {
    [FAMILY_MINIMAX]: [MODE_I2VA, MODE_FL2VA, MODE_R2VA],
    [FAMILY_WAN]: [MODE_I2V, MODE_T2V],
    [FAMILY_LTX]: [MODE_I2V, MODE_FL2VA],
};

const DEFAULT_FAMILY_MODE = {
    [FAMILY_MINIMAX]: MODE_I2VA,
    [FAMILY_WAN]: MODE_I2V,
    [FAMILY_LTX]: MODE_I2V,
};

// Keep in sync with MODE_FAMILY_ANALOG in AILab_QwenVL_Finetune.py
const MODE_FAMILY_ANALOG = {
    [MODE_I2VA]: { [FAMILY_MINIMAX]: MODE_I2VA, [FAMILY_WAN]: MODE_I2V, [FAMILY_LTX]: MODE_I2V },
    [MODE_I2V]: { [FAMILY_MINIMAX]: MODE_I2VA, [FAMILY_WAN]: MODE_I2V, [FAMILY_LTX]: MODE_I2V },
    [MODE_T2V]: { [FAMILY_MINIMAX]: MODE_I2VA, [FAMILY_WAN]: MODE_T2V, [FAMILY_LTX]: MODE_I2V },
    [MODE_FL2VA]: { [FAMILY_MINIMAX]: MODE_FL2VA, [FAMILY_WAN]: MODE_I2V, [FAMILY_LTX]: MODE_FL2VA },
    [MODE_R2VA]: { [FAMILY_MINIMAX]: MODE_R2VA, [FAMILY_WAN]: MODE_I2V, [FAMILY_LTX]: MODE_I2V },
};

const TARGET_TO_FAMILY_MODE = {
    [FAMILY_CUSTOM]: { family: FAMILY_CUSTOM, mode: INHERIT },
    "🎬 MiniMax H3 I2VA": { family: FAMILY_MINIMAX, mode: MODE_I2VA },
    "🔄 MiniMax H3 FL2VA": { family: FAMILY_MINIMAX, mode: MODE_FL2VA },
    "🎞️ MiniMax H3 R2VA": { family: FAMILY_MINIMAX, mode: MODE_R2VA },
    "🎥 Wan 2.2 I2V": { family: FAMILY_WAN, mode: MODE_I2V },
    "🍿 Wan 2.2 T2V": { family: FAMILY_WAN, mode: MODE_T2V },
    "🎥 LTX 2.3 I2V": { family: FAMILY_LTX, mode: MODE_I2V },
    "🔀 LTX 2.3 FL2VA": { family: FAMILY_LTX, mode: MODE_FL2VA },
};

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget && widget.name === name);
}

function setComboValues(widget, values) {
    if (!widget) {
        return;
    }
    widget.options = widget.options || {};
    widget.options.values = values;
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
    widget.type = widget._qwenOrigType || "combo";
    widget.hidden = false;
    if (widget.computeSize) {
        delete widget.computeSize;
    }
}

function defaultModeForFamily(family) {
    if (!family || family === INHERIT || family === FAMILY_CUSTOM) {
        return INHERIT;
    }
    return DEFAULT_FAMILY_MODE[family] || INHERIT;
}

function compatibleMode(family, mode) {
    const allowed = FAMILY_MODES[family];
    if (!allowed || !allowed.length) {
        return INHERIT;
    }
    if (allowed.includes(mode)) {
        return mode;
    }
    const analog = MODE_FAMILY_ANALOG[mode]?.[family];
    if (analog && allowed.includes(analog)) {
        return analog;
    }
    return allowed[0];
}

function splitTarget(value) {
    if (!value || value === INHERIT) {
        return { family: INHERIT, mode: INHERIT, fromTarget: false };
    }
    const mapped = TARGET_TO_FAMILY_MODE[value];
    if (mapped) {
        return { ...mapped, fromTarget: value !== mapped.family };
    }
    if (FAMILIES.includes(value)) {
        return { family: value, mode: defaultModeForFamily(value), fromTarget: false };
    }
    return { family: value, mode: INHERIT, fromTarget: false };
}

function migratePromptOptionsValues(node, info) {
    const values = info?.widgets_values ?? node.widgets_values;
    if (Array.isArray(values) && values.length >= 4 && typeof values[1] === "number") {
        const split = splitTarget(values[0]);
        const mode = split.fromTarget ? split.mode : defaultModeForFamily(split.family);
        const next = [split.family, mode, values[1], values[2], values[3]];
        if (info && Array.isArray(info.widgets_values)) {
            info.widgets_values = next;
        }
        node.widgets_values = next;
        return;
    }
    if (values && !Array.isArray(values) && values.target && values.family == null) {
        const split = splitTarget(values.target);
        values.family = split.family;
        values.mode = split.fromTarget ? split.mode : defaultModeForFamily(split.family);
    }
}

function wrapCallback(widget, fn) {
    if (!widget || widget._qwenTargetWrapped) {
        return;
    }
    widget._qwenTargetWrapped = true;
    const original = widget.callback;
    widget.callback = function () {
        const result = original?.apply(this, arguments);
        fn();
        return result;
    };
}

function syncTargetWidgets(node) {
    if (!node || node._qwenTargetSync) {
        return;
    }
    node._qwenTargetSync = true;
    try {
        const familyWidget = findWidget(node, "family");
        const modeWidget = findWidget(node, "mode");
        if (!familyWidget || !modeWidget) {
            return;
        }

        wrapCallback(familyWidget, () => syncTargetWidgets(node));
        wrapCallback(modeWidget, () => syncTargetWidgets(node));

        const split = splitTarget(familyWidget.value);
        if (split.fromTarget) {
            familyWidget.value = split.family;
            modeWidget.value = split.mode;
        }

        setComboValues(familyWidget, FAMILIES);
        if (!FAMILIES.includes(familyWidget.value)) {
            familyWidget.value = split.family;
        }

        const family = familyWidget.value;
        const showMode = Boolean(FAMILY_MODES[family]?.length);
        if (showMode) {
            const allowed = FAMILY_MODES[family];
            setComboValues(modeWidget, allowed);
            const chosen = compatibleMode(family, modeWidget.value);
            if (modeWidget.value !== chosen) {
                modeWidget.value = chosen;
            }
            setWidgetHidden(modeWidget, false);
        } else {
            setWidgetHidden(modeWidget, true);
        }

        if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
            node.setSize(node.computeSize());
        }
        node.graph?.setDirtyCanvas?.(true, true);
    } finally {
        node._qwenTargetSync = false;
    }
}

app.registerExtension({
    name: "QwenVL.dynamicTarget",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            syncTargetWidgets(this);
            setTimeout(() => syncTargetWidgets(this), 0);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            migratePromptOptionsValues(this, info);
            const result = onConfigure?.apply(this, arguments);
            syncTargetWidgets(this);
            setTimeout(() => syncTargetWidgets(this), 0);
            return result;
        };
    },
});
