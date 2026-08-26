# QwenVL-God Options
# Chainable option bags. Each node upserts its set fields onto an incoming
# options dict (or starts a new one) and outputs the merged result.
# These nodes only expose settings that are not widgets on QwenVL-God Finetune.

from AILab_QwenVL import TOOLTIPS

QWENVL_OPTIONS = "QWENVL_OPTIONS"
INHERIT = "inherit"
DEBUG_KEY = "_debug"
DEBUG_PREVIEW_CHARS = 160

OPTIONS_PORT = {
    "options": (QWENVL_OPTIONS, {
        "tooltip": "Incoming options bag. This node upserts any fields set above and outputs the merge.",
    }),
}

OPTIONS_RETURN_TYPES = (QWENVL_OPTIONS, "STRING")
OPTIONS_RETURN_NAMES = ("options", "debug")
GOD_CATEGORY = "QwenVL-God"
BOOL_CHOICES = [INHERIT, "true", "false"]


def is_unset(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in ("", INHERIT):
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)) and value < 0:
        return True
    return False


def merge_options(base, updates: dict) -> dict:
    merged = dict(base) if isinstance(base, dict) else {}
    for key, value in (updates or {}).items():
        if key == DEBUG_KEY or is_unset(value):
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            nested = dict(merged[key])
            for nested_key, nested_value in value.items():
                if not is_unset(nested_value):
                    nested[nested_key] = nested_value
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def debug_steps(options) -> list:
    if not isinstance(options, dict):
        return []
    steps = options.get(DEBUG_KEY)
    return list(steps) if isinstance(steps, list) else []


def public_keys(options) -> list[str]:
    if not isinstance(options, dict):
        return []
    return [key for key in options if key != DEBUG_KEY]


def _preview(value, limit=DEBUG_PREVIEW_CHARS):
    if isinstance(value, dict):
        return {key: _preview(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_preview(item, limit) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… ({len(text)} chars)"


def _applied_fields(updates: dict) -> dict:
    applied = {}
    for key, value in (updates or {}).items():
        if key == DEBUG_KEY or is_unset(value):
            continue
        applied[key] = value
    return applied


def _format_value_lines(value, indent: str) -> list[str]:
    if isinstance(value, dict):
        if not value:
            return [f"{indent}{{}}"]
        lines = []
        for key, item in value.items():
            item_lines = _format_value_lines(item, indent + "  ")
            if isinstance(item, dict):
                lines.append(f"{indent}{key}:")
                lines.extend(item_lines)
            else:
                lines.append(f"{indent}{key} = {item_lines[0].lstrip()}")
                lines.extend(item_lines[1:])
        return lines
    if isinstance(value, list):
        return [f"{indent}{value}"]
    if isinstance(value, str) and "\n" in value:
        first, *rest = value.splitlines()
        lines = [f"{indent}{first}"]
        inner = indent + "  "
        lines.extend(f"{inner}{line}" for line in rest)
        return lines
    return [f"{indent}{value}"]


def _source_label(options, key) -> str:
    if not isinstance(options, dict) or key not in options or is_unset(options[key]):
        return "default"
    return "options"


def format_model_input_preview(options, custom_prompt=None, extra_tokens=None) -> str:
    try:
        from AILab_QwenVL_Finetune import (
            PROMPT_OPTION_DEFAULTS,
            pick_prepend_system_prompt,
            resolve_finetune_messages,
            stringify_token,
        )
        from AILab_QwenVL_MiniMax_Keywords import (
            H3_KEYWORD_PLACEMENT_KEY,
            PLACEMENT_BOTH,
            PLACEMENT_WEAVE,
            attach_h3_keyword_instructions,
            collect_h3_keywords,
            collect_h3_overrides,
            remap_visual_keywords,
            remap_visual_overrides,
        )
    except Exception as exc:
        return f"(prompt preview unavailable: {exc})"

    cfg = resolve_options(options, PROMPT_OPTION_DEFAULTS)
    cfg["prepend_system_prompt"] = pick_prepend_system_prompt(
        options, cfg["prepend_system_prompt"]
    )
    target = cfg["target"]
    extra_tokens = extra_tokens or {}
    user_custom = custom_prompt if custom_prompt is not None else ""
    try:
        system_prompt, user_prompt = resolve_finetune_messages(
            target=target,
            custom_prompt=user_custom,
            prepend_system_prompt=cfg["prepend_system_prompt"],
            duration_s=cfg["duration_seconds"],
            segment_s=cfg["segment_seconds"],
            layout=cfg["prompt_layout"],
            extra_tokens=extra_tokens,
        )
    except Exception as exc:
        return f"(prompt preview not ready: {exc})"

    keywords = remap_visual_keywords(collect_h3_keywords(options), target)
    overrides = remap_visual_overrides(collect_h3_overrides(options), target)
    placement = pick_option(options, H3_KEYWORD_PLACEMENT_KEY, PLACEMENT_BOTH)
    system_prompt, user_prompt = attach_h3_keyword_instructions(
        system_prompt, user_prompt, keywords, placement, overrides
    )

    lines = [
        "resulting model input",
        "=====================",
        "resolved prompt options",
        "-----------------------",
    ]
    for key, value in cfg.items():
        shown = value if str(value).strip() else "(empty)"
        lines.append(f"{key}: {shown} ({_source_label(options, key)})")
    if custom_prompt is not None:
        shown = custom_prompt if str(custom_prompt).strip() else "(empty)"
        lines.append(f"custom_prompt: {shown} (widget)")
    if extra_tokens:
        lines.append("finetune extra_tokens (custom_prompt):")
        for key, value in extra_tokens.items():
            lines.append(f"  {{{key}}} = {stringify_token(value)}")
    lines.extend([
        "",
        "SYSTEM",
        "------",
        system_prompt or "(empty)",
        "",
        "USER",
        "----",
        user_prompt or "(empty)",
        "",
        "post-generation",
        "---------------",
    ])
    if overrides:
        lines.append("section overrides will replace:")
        for section, value in overrides.items():
            lines.append(f"  {section}: {value}")
    else:
        lines.append("section overrides: (none)")
    kept = {key: tokens for key, tokens in keywords.items() if key not in overrides}
    if kept and placement != PLACEMENT_WEAVE:
        lines.append(f"missing keyword phrases will be appended (placement={placement}):")
        for section, tokens in kept.items():
            lines.append(f"  {section}: {', '.join(tokens)}")
    elif kept:
        lines.append(f"keywords instructed only, not appended (placement={placement})")
    else:
        lines.append("keyword append: (none)")
    return "\n".join(lines)


def format_debug_log(options, custom_prompt=None, extra_tokens=None) -> str:
    steps = debug_steps(options)
    if not steps:
        path = "(no options path)"
    else:
        lines = []
        for index, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                lines.append(f"{index}. {step}")
                continue
            lines.append(f"{index}. {step.get('node', '?')}")
            applied = step.get("set") or {}
            if applied:
                lines.append("   set:")
                lines.extend(_format_value_lines(_preview(applied), "     "))
            elif not step.get("notes"):
                lines.append("   set: (nothing; all fields inherit/empty)")
            for note in step.get("notes") or []:
                for line in str(note).splitlines() or [""]:
                    lines.append(f"   {line}")
            bag_keys = step.get("bag_keys")
            if bag_keys:
                lines.append("   bag keys: " + ", ".join(bag_keys))
        path = "\n".join(lines)
    preview = format_model_input_preview(
        options, custom_prompt=custom_prompt, extra_tokens=extra_tokens
    )
    return path + "\n\n" + preview


def record_options_step(base, updates, node_name: str, notes=None) -> dict:
    merged = merge_options(base, updates)
    step = {
        "node": node_name,
        "set": _applied_fields(updates),
        "notes": [note for note in (notes or []) if note],
        "bag_keys": public_keys(merged),
    }
    merged[DEBUG_KEY] = debug_steps(base) + [step]
    return merged


def options_result(base, updates, node_name: str, notes=None) -> tuple:
    merged = record_options_step(base, updates, node_name, notes)
    return (merged, format_debug_log(merged))


def pick_option(options, key, default):
    if not isinstance(options, dict) or key not in options:
        return default
    value = options[key]
    if is_unset(value):
        return default
    return value


def resolve_options(options, defaults: dict) -> dict:
    return {key: pick_option(options, key, default) for key, default in defaults.items()}


def coerce_bool_choice(value):
    if value is None or (isinstance(value, str) and value.strip() in ("", INHERIT)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


class AILab_QwenVL_Prompt_Options:
    @classmethod
    def INPUT_TYPES(cls):
        from AILab_QwenVL_Finetune import (
            FINETUNE_FAMILIES,
            FINETUNE_MODES,
            FINETUNE_TARGETS,
            PREPEND_SYSTEM_PROMPT_TOKEN_HELP,
            VIDEO_PROMPT_EXTRA_TOKENS_TOOLTIP,
        )

        family_choices = [INHERIT]
        for item in (*FINETUNE_FAMILIES, *FINETUNE_TARGETS):
            if item not in family_choices:
                family_choices.append(item)
        mode_choices = [INHERIT, *FINETUNE_MODES]
        return {
            "required": {
                "family": (family_choices, {
                    "default": INHERIT,
                    "tooltip": (
                        "Video model family. inherit leaves target unset so an upstream "
                        "Options node or the Finetune default is used."
                    ),
                }),
                "mode": (mode_choices, {
                    "default": INHERIT,
                    "tooltip": (
                        "Prompt type for the selected family. Switching family keeps a "
                        "compatible mode when possible (I2VA↔I2V; FL2VA stays FL2VA if "
                        "the new family has it). Ignored when family is inherit or Custom Only."
                    ),
                }),
                "duration_seconds": ("INT", {"default": -1, "min": -1, "max": 60, "tooltip": "Clip length in seconds. -1 = inherit / unset. Available in prompts as {duration_seconds} or {duration_s}."}),
                "segment_seconds": ("INT", {"default": -1, "min": -1, "max": 20, "tooltip": "Wan clip chunk length. -1 = inherit / unset. Available in prompts as {segment_seconds} or {segment_s}."}),
                "prepend_system_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prepended to the target system prompt. Empty = inherit / unset. " + PREPEND_SYSTEM_PROMPT_TOKEN_HELP}),
            },
            "optional": {
                **OPTIONS_PORT,
                "extra_tokens": ("DICT", {"tooltip": VIDEO_PROMPT_EXTRA_TOKENS_TOOLTIP}),
            },
        }

    RETURN_TYPES = (QWENVL_OPTIONS, "STRING", "INT", "INT")
    RETURN_NAMES = ("options", "debug", "duration_seconds", "segment_seconds")
    FUNCTION = "build"
    CATEGORY = GOD_CATEGORY

    def build(
        self,
        family,
        mode,
        duration_seconds,
        segment_seconds,
        prepend_system_prompt,
        options=None,
        extra_tokens=None,
        **kwargs,
    ):
        from AILab_QwenVL_Finetune import (
            PROMPT_OPTION_DEFAULTS,
            apply_tokens,
            coerce_token_dict,
            resolve_target_from_family_mode,
            stringify_token,
        )
        extra = coerce_token_dict(extra_tokens)
        prepend = prepend_system_prompt
        notes = []
        if extra:
            if is_unset(prepend):
                notes.append("extra_tokens unused: prepend_system_prompt is empty")
            else:
                prepend = apply_tokens(str(prepend), extra)
                notes.append("extra_tokens applied to prepend_system_prompt:")
                for key, value in extra.items():
                    notes.append(f"  {{{key}}} = {stringify_token(value)}")
        legacy_target = kwargs.get("target")
        if is_unset(family) and not is_unset(legacy_target):
            family = legacy_target
        target, mode_note = resolve_target_from_family_mode(family, mode)
        if mode_note:
            notes.append(mode_note)
        own = {
            "target": target if target is not None else INHERIT,
            "duration_seconds": duration_seconds,
            "segment_seconds": segment_seconds,
            "prepend_system_prompt": prepend,
        }
        merged, debug = options_result(options, own, "QwenVL-God Options (Video Prompt)", notes)
        duration = int(pick_option(merged, "duration_seconds", PROMPT_OPTION_DEFAULTS["duration_seconds"]))
        segment = int(pick_option(merged, "segment_seconds", PROMPT_OPTION_DEFAULTS["segment_seconds"]))
        return (merged, debug, duration, segment)


class AILab_QwenVL_Generation_Options:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_tokens": ("INT", {"default": -1, "min": -1, "max": 8192, "tooltip": TOOLTIPS["max_tokens"] + " -1 = inherit / unset."}),
                "temperature": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": TOOLTIPS["temperature"] + " -1 = inherit / unset."}),
                "top_p": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": TOOLTIPS["top_p"] + " -1 = inherit / unset."}),
                "num_beams": ("INT", {"default": -1, "min": -1, "max": 8, "tooltip": TOOLTIPS["num_beams"] + " -1 = inherit / unset."}),
                "repetition_penalty": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, "tooltip": TOOLTIPS["repetition_penalty"] + " -1 = inherit / unset."}),
                "frame_count": ("INT", {"default": -1, "min": -1, "max": 64, "tooltip": TOOLTIPS["frame_count"] + " -1 = inherit / unset."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"] + " -1 = inherit / unset."}),
            },
            "optional": OPTIONS_PORT,
        }

    RETURN_TYPES = OPTIONS_RETURN_TYPES
    RETURN_NAMES = OPTIONS_RETURN_NAMES
    FUNCTION = "build"
    CATEGORY = GOD_CATEGORY

    def build(
        self,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        frame_count,
        seed,
        options=None,
    ):
        own = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "frame_count": frame_count,
            "seed": seed,
        }
        return options_result(options, own, "QwenVL-God Options (Generation)")


class AILab_QwenVL_Wan_Options:
    @classmethod
    def INPUT_TYPES(cls):
        from AILab_QwenVL_Finetune import LAYOUT_AUTO, LAYOUT_SCENE, LAYOUT_TIMELINE

        inherit_layouts = [INHERIT, LAYOUT_AUTO, LAYOUT_TIMELINE, LAYOUT_SCENE]
        return {
            "required": {
                "prompt_layout": (inherit_layouts, {
                    "default": INHERIT,
                    "tooltip": (
                        "Wan 2.2 only. How the prompt is shaped. "
                        "timeline = (At 0 seconds: ...) markers for each second. "
                        "scene = flowing paragraph(s), no timestamps. "
                        "auto = timeline. "
                        "If duration is longer than segment_seconds, you get one block per Wan clip. "
                        "MiniMax and LTX ignore this except as {layout} in custom_prompt. inherit = unset."
                    ),
                }),
            },
            "optional": OPTIONS_PORT,
        }

    RETURN_TYPES = OPTIONS_RETURN_TYPES
    RETURN_NAMES = OPTIONS_RETURN_NAMES
    FUNCTION = "build"
    CATEGORY = GOD_CATEGORY

    def build(self, prompt_layout, options=None):
        return options_result(
            options,
            {"prompt_layout": prompt_layout},
            "QwenVL-God Options (Wan 2.2)",
        )


class AILab_QwenVL_Runtime_Options:
    @classmethod
    def INPUT_TYPES(cls):
        import torch
        from AILab_QwenVL import ATTENTION_MODES, Quantization

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = [INHERIT, "auto", "cpu", "mps"] + gpu_list
        return {
            "required": {
                "quantization": ([INHERIT, *Quantization.get_values()], {"default": INHERIT, "tooltip": TOOLTIPS["quantization"] + " inherit = unset."}),
                "attention_mode": ([INHERIT, *ATTENTION_MODES], {"default": INHERIT, "tooltip": TOOLTIPS["attention_mode"] + " inherit = unset."}),
                "device": (device_options, {"default": INHERIT, "tooltip": TOOLTIPS["device"] + " inherit = unset."}),
                "use_torch_compile": (BOOL_CHOICES, {"default": INHERIT, "tooltip": TOOLTIPS["use_torch_compile"] + " inherit = unset."}),
                "keep_model_loaded": (BOOL_CHOICES, {"default": INHERIT, "tooltip": TOOLTIPS["keep_model_loaded"] + " inherit = unset."}),
                "keep_last_prompt": (BOOL_CHOICES, {"default": INHERIT, "tooltip": "Keep last generated prompt instead of creating a new one. inherit = unset."}),
                "dry_run": (BOOL_CHOICES, {"default": INHERIT, "tooltip": "Build prompts and debug without loading or running the model. inherit = unset."}),
            },
            "optional": OPTIONS_PORT,
        }

    RETURN_TYPES = OPTIONS_RETURN_TYPES
    RETURN_NAMES = OPTIONS_RETURN_NAMES
    FUNCTION = "build"
    CATEGORY = GOD_CATEGORY

    def build(
        self,
        quantization,
        attention_mode,
        device,
        use_torch_compile,
        keep_model_loaded,
        keep_last_prompt,
        dry_run,
        options=None,
    ):
        own = {
            "quantization": quantization,
            "attention_mode": attention_mode,
            "device": device,
        }
        for key, value in {
            "use_torch_compile": use_torch_compile,
            "keep_model_loaded": keep_model_loaded,
            "keep_last_prompt": keep_last_prompt,
            "dry_run": dry_run,
        }.items():
            parsed = coerce_bool_choice(value)
            own[key] = parsed if parsed is not None else INHERIT
        return options_result(options, own, "QwenVL-God Options (Runtime)")


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_Prompt_Options": AILab_QwenVL_Prompt_Options,
    "AILab_QwenVL_Generation_Options": AILab_QwenVL_Generation_Options,
    "AILab_QwenVL_Wan_Options": AILab_QwenVL_Wan_Options,
    "AILab_QwenVL_Runtime_Options": AILab_QwenVL_Runtime_Options,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_Prompt_Options": "QwenVL-God Options (Video Prompt)",
    "AILab_QwenVL_Generation_Options": "QwenVL-God Options (Generation)",
    "AILab_QwenVL_Wan_Options": "QwenVL-God Options (Wan 2.2)",
    "AILab_QwenVL_Runtime_Options": "QwenVL-God Options (Runtime)",
}
