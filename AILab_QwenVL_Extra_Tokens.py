# QwenVL-God Extra Tokens
# Builds a DICT of {name} replacements from dynamically wired values.
# Each connected value gets a one-line key widget; unused slots stay hidden.

MAX_TOKEN_SLOTS = 16


class AnyType(str):
    """Wildcard slot type so ComfyUI will accept any connected output."""

    def __eq__(self, _other) -> bool:
        return True

    def __ne__(self, _other) -> bool:
        return False


ANY_TYPE = AnyType("*")

KEY_TOOLTIP = (
    "Name for the matching value# socket, used as {name} in the prompt field "
    "of the node this extra_tokens output is connected to. Identifier only: "
    "start with a letter or underscore."
)
VALUE_TOOLTIP = (
    "Connect any value. When wired, a key widget appears for this slot and "
    "the next value socket is added. The original type is kept; it is "
    "stringified only when substituted into prompt text."
)


def extra_token_input_types(max_slots=MAX_TOKEN_SLOTS):
    optional = {}
    for index in range(1, max_slots + 1):
        optional[f"value{index}"] = (ANY_TYPE, {"tooltip": VALUE_TOOLTIP})
    for index in range(1, max_slots + 1):
        optional[f"key{index}"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "placeholder": "token name",
                "tooltip": KEY_TOOLTIP,
            },
        )
    optional["extra_tokens"] = (
        "DICT",
        {"tooltip": "Optional. Merge another Extra Tokens node so bags can be chained."},
    )
    return optional


def normalize_token_key(name) -> str:
    key = str(name or "").strip()
    if len(key) >= 2 and key.startswith("{") and key.endswith("}"):
        key = key[1:-1].strip()
    return key


def collect_extra_token_slots(extra_tokens=None, max_slots=MAX_TOKEN_SLOTS, **kwargs) -> dict:
    from AILab_QwenVL_Finetune import TOKEN_NAME_RE, coerce_token_dict

    merged = dict(coerce_token_dict(extra_tokens))
    seen = {}
    for index in range(1, max_slots + 1):
        value_key = f"value{index}"
        if value_key not in kwargs:
            continue
        value = kwargs.get(value_key)
        if value is None:
            continue
        name = normalize_token_key(kwargs.get(f"key{index}"))
        if not name:
            raise ValueError(
                f"extra token slot {index} is connected but its key widget is empty. "
                "Type a name like subject to use as {subject}."
            )
        if not TOKEN_NAME_RE.match(name):
            raise ValueError(
                f"Invalid extra token key {name!r} on slot {index}. "
                "Use a letter or underscore first, then letters, digits, or underscores."
            )
        if name in seen:
            raise ValueError(
                f"Duplicate extra token key '{name}' on slots {seen[name]} and {index}."
            )
        seen[name] = index
        merged[name] = value
    return merged


class AILab_QwenVL_Extra_Tokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": extra_token_input_types(),
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("extra_tokens",)
    FUNCTION = "build"
    CATEGORY = "QwenVL-God"
    DESCRIPTION = (
        "Dynamic {name} token bag. Wire a value#, type its name in the matching "
        "widget, and connect extra_tokens to Video Prompt (prepend_system_prompt) "
        "or Finetune (custom_prompt)."
    )

    def build(self, extra_tokens=None, **kwargs):
        return (collect_extra_token_slots(extra_tokens, **kwargs),)


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_Extra_Tokens": AILab_QwenVL_Extra_Tokens,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_Extra_Tokens": "QwenVL-God Extra Tokens",
}
