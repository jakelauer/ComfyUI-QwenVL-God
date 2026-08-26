import { app } from "/scripts/app.js";

const DYNAMIC_IMAGE_NODES = {
    AILab_QwenVL: "image2",
    AILab_QwenVL_Advanced: "image2",
    AILab_QwenVL_Finetune: "image2",
    AILab_QwenVL_GGUF: "video",
    AILab_QwenVL_GGUF_Advanced: "video",
};

const MAX_IMAGE_INDEX = 16;

function imageIndex(name) {
    if (name === "image") {
        return 1;
    }
    if (name === "image2" || name === "video") {
        return 2;
    }
    const match = /^image(\d+)$/.exec(name || "");
    return match ? parseInt(match[1], 10) : null;
}

function addImageSlot(node, name) {
    node.addInput(name, "IMAGE");
    const added = node.inputs.pop();
    const optionsIdx = node.inputs.findIndex((input) => input && input.name === "options");
    if (optionsIdx >= 0) {
        node.inputs.splice(optionsIdx, 0, added);
    } else {
        node.inputs.push(added);
    }
}

function syncDynamicImageInputs(node, secondName) {
    if (!node || !node.inputs || node._qwenImageSync) {
        return;
    }
    node._qwenImageSync = true;
    try {
        let highestConnected = 0;
        for (const input of node.inputs) {
            const index = imageIndex(input?.name);
            if (index != null && input.link != null) {
                highestConnected = Math.max(highestConnected, index);
            }
        }
        const wantMax = Math.min(MAX_IMAGE_INDEX, Math.max(2, highestConnected + 1));

        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const index = imageIndex(node.inputs[i]?.name);
            if (index != null && index >= 3 && index > wantMax && node.inputs[i].link == null) {
                node.removeInput(i);
            }
        }

        for (let n = 3; n <= wantMax; n++) {
            const name = `image${n}`;
            if (!node.inputs.some((input) => input && input.name === name)) {
                addImageSlot(node, name);
            }
        }

        if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
            node.setSize(node.computeSize());
        }
        node.graph?.setDirtyCanvas?.(true, true);
    } finally {
        node._qwenImageSync = false;
    }
}

app.registerExtension({
    name: "QwenVL.dynamicImageInputs",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const secondName = DYNAMIC_IMAGE_NODES[nodeData?.name];
        if (!secondName) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            setTimeout(() => syncDynamicImageInputs(this, secondName), 0);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => syncDynamicImageInputs(this, secondName), 0);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            const result = onConnectionsChange?.apply(this, arguments);
            if (type === 1) {
                syncDynamicImageInputs(this, secondName);
            }
            return result;
        };
    },
});
