# MiniMax H3 LoRA / trigger keyword Options node.
# Upserts per-section keyword lists (or full section overrides) onto a
# QWENVL_OPTIONS bag so Finetune can weave/append phrases or replace sections.

import re

from AILab_QwenVL_Options import OPTIONS_RETURN_NAMES, OPTIONS_RETURN_TYPES, QWENVL_OPTIONS, options_result

H3_KEYWORD_PREFIX = "h3_keywords_"
H3_KEYWORD_PLACEMENT_KEY = "h3_keyword_placement"
H3_OVERRIDE_KEY = "h3_section_overrides"

H3_KEYWORD_SECTIONS = [
    "visual",
    "integrated_multimodal_description",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
    "subject_definitions",
    "summary",
    "retention_analysis",
]

H3_OUTPUT_SECTIONS = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
]

PLACEMENT_BOTH = "both"
PLACEMENT_WEAVE = "weave"
PLACEMENT_APPEND = "append"

_SECTION_HEADER_RE = re.compile(
    r"^(" + "|".join(H3_OUTPUT_SECTIONS) + r")\s*:",
    re.MULTILINE,
)


def parse_keyword_field(text) -> list[str]:
    tokens = []
    for line in str(text or "").splitlines():
        for part in line.split(","):
            token = part.strip()
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def collect_h3_keywords(options) -> dict[str, list[str]]:
    if not isinstance(options, dict):
        return {}
    collected = {}
    for section in H3_KEYWORD_SECTIONS:
        tokens = parse_keyword_field(options.get(H3_KEYWORD_PREFIX + section))
        if tokens:
            collected[section] = tokens
    return collected


def _visual_dest(target: str) -> str:
    target_l = (target or "").lower()
    if "r2va" in target_l:
        return "detailed_description"
    if "minimax" in target_l:
        return "integrated_multimodal_description"
    return "_prompt"


def remap_visual_keywords(keywords: dict[str, list[str]], target: str) -> dict[str, list[str]]:
    remapped = {key: list(value) for key, value in keywords.items()}
    visual = remapped.pop("visual", None)
    if not visual:
        return remapped
    dest = _visual_dest(target)
    existing = remapped.setdefault(dest, [])
    for token in visual:
        if token not in existing:
            existing.append(token)
    return remapped


def collect_h3_overrides(options) -> dict[str, str]:
    if not isinstance(options, dict):
        return {}
    raw = options.get(H3_OVERRIDE_KEY)
    if not isinstance(raw, dict):
        return {}
    return {
        key: value.strip()
        for key, value in raw.items()
        if isinstance(value, str) and value.strip()
    }


def remap_visual_overrides(sections: dict[str, str], target: str) -> dict[str, str]:
    remapped = dict(sections)
    visual = remapped.pop("visual", None)
    if not visual:
        return remapped
    dest = _visual_dest(target)
    remapped.setdefault(dest, visual)
    return remapped


def attach_h3_keyword_instructions(
    system_prompt: str,
    user_prompt: str,
    keywords: dict[str, list[str]],
    placement: str,
    overrides: dict[str, str] | None = None,
) -> tuple[str, str]:
    overrides = overrides or {}
    if overrides:
        skip = ", ".join(f"`{name}`" for name in overrides if name != "_prompt")
        if "_prompt" in overrides:
            override_rules = (
                "SECTION OVERRIDES:\n"
                "The entire output will be replaced after generation with provided text. "
                "Keep the reply short."
            )
        else:
            override_rules = (
                "SECTION OVERRIDES:\n"
                "The following sections will be replaced after generation with exact provided text. "
                "You may omit them or emit a one-line stub. Focus on the remaining sections.\n"
                f"- {skip}"
            )
        system_prompt = "\n\n".join(part for part in (system_prompt, override_rules) if part)

    keywords = {key: value for key, value in keywords.items() if key not in overrides}
    if not keywords:
        return system_prompt, user_prompt

    lines = []
    user_lines = []
    for section, tokens in keywords.items():
        label = "the output prompt" if section == "_prompt" else f"`{section}`"
        quoted = "; ".join(f"`{token}`" for token in tokens)
        lines.append(f"- {label}: {quoted}")
        user_lines.append(f"{section}: {', '.join(tokens)}")

    if placement == PLACEMENT_APPEND:
        rules = (
            "LORA TRIGGER KEYWORDS:\n"
            "The listed phrases are LoRA / embedding triggers. Prefer including them "
            "verbatim in the named sections. If omitted, they will be appended after generation.\n"
            + "\n".join(lines)
        )
    else:
        rules = (
            "LORA TRIGGER KEYWORDS — MANDATORY:\n"
            "The following phrases are LoRA / embedding trigger tokens. They MUST appear "
            "verbatim (exact spelling, spacing, and punctuation) in the named output sections. "
            "Do not translate, paraphrase, correct, or split them. Weave each phrase into the "
            "prose of that section where it reads naturally. If a phrase cannot be woven in, "
            "append it at the end of that section.\n"
            + "\n".join(lines)
        )

    user_block = "[LORA KEYWORDS]\n" + "\n".join(user_lines)
    system_prompt = "\n\n".join(part for part in (system_prompt, rules) if part)
    user_prompt = "\n\n".join(part for part in (user_prompt, user_block) if part)
    return system_prompt, user_prompt


def ensure_h3_keywords(text: str, keywords: dict[str, list[str]], placement: str) -> str:
    if not keywords or placement == PLACEMENT_WEAVE:
        return text
    text = text or ""

    matches = list(_SECTION_HEADER_RE.finditer(text))
    found = {match.group(1) for match in matches}
    # Edit from the end so earlier indices stay valid.
    for index in range(len(matches) - 1, -1, -1):
        match = matches[index]
        name = match.group(1)
        tokens = keywords.get(name) or []
        if not tokens:
            continue
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        missing = [token for token in tokens if token not in body]
        if not missing:
            continue
        insert_at = body_start + len(body.rstrip())
        text = text[:insert_at] + " " + ", ".join(missing) + text[insert_at:]

    extras = []
    for name, tokens in keywords.items():
        if name == "_prompt" or name in found or not tokens:
            continue
        extras.append(f"{name}: {', '.join(tokens)}")
    if extras:
        text = text.rstrip() + "\n\n" + "\n".join(extras)

    generic = keywords.get("_prompt") or []
    missing_generic = [token for token in generic if token not in text]
    if missing_generic:
        text = text.rstrip() + "\n" + ", ".join(missing_generic)
    return text


def apply_h3_section_overrides(text: str, sections: dict[str, str]) -> str:
    if not sections:
        return text
    text = text or ""
    if "_prompt" in sections:
        named = {key: value for key, value in sections.items() if key != "_prompt"}
        text = sections["_prompt"].strip()
        if not named:
            return text
        sections = named

    matches = list(_SECTION_HEADER_RE.finditer(text))
    found = {match.group(1) for match in matches}
    for index in range(len(matches) - 1, -1, -1):
        match = matches[index]
        name = match.group(1)
        value = sections.get(name)
        if not value:
            continue
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        replacement = "\n" + value.strip() + "\n"
        text = text[:body_start] + replacement + text[body_end:]

    extras = []
    for name, value in sections.items():
        if name == "_prompt" or name in found or not value:
            continue
        extras.append(f"{name}: {value.strip()}")
    if extras:
        text = text.rstrip() + "\n\n" + "\n".join(extras)
    return text


class AILab_QwenVL_MiniMax_Keywords:
    @classmethod
    def INPUT_TYPES(cls):
        section_tooltip = (
            "When override is off: LoRA / embedding trigger phrases (comma or newline separated). "
            "When override is on: the full replacement text for this section. Empty = unset."
        )
        return {
            "required": {
                "visual": ("STRING", {"default": "", "multiline": True, "tooltip": "Convenience field. Routed to integrated_multimodal_description (H3/FL2VA), detailed_description (R2VA), or the whole prompt otherwise."}),
                "integrated_multimodal_description": ("STRING", {"default": "", "multiline": True, "tooltip": section_tooltip}),
                "detailed_description": ("STRING", {"default": "", "multiline": True, "tooltip": section_tooltip + " Used by R2VA."}),
                "overall_soundscape": ("STRING", {"default": "", "multiline": True, "tooltip": section_tooltip}),
                "non_diegetic_music": ("STRING", {"default": "", "multiline": True, "tooltip": section_tooltip}),
                "subject_definitions": ("STRING", {"default": "", "multiline": True, "tooltip": section_tooltip + " Used by R2VA."}),
                "summary": ("STRING", {"default": "", "multiline": True, "tooltip": section_tooltip + " Used by R2VA."}),
                "retention_analysis": ("STRING", {"default": "", "multiline": True, "tooltip": section_tooltip + " Used by R2VA."}),
                "placement": ([PLACEMENT_BOTH, PLACEMENT_WEAVE, PLACEMENT_APPEND], {
                    "default": PLACEMENT_BOTH,
                    "tooltip": (
                        "How keyword phrases are applied. Ignored when override is on. "
                        "weave: tell the model to include each phrase naturally in its section; nothing is added after generation. "
                        "append: do not instruct the model; after generation, add any missing phrases to the matching sections. "
                        "both: weave during generation, then append any phrases that are still missing."
                    ),
                }),
                "override": ("BOOLEAN", {"default": False, "tooltip": "When enabled, filled section fields fully replace those sections in the finished prompt instead of being treated as keywords."}),
            },
            "optional": {
                "options": (QWENVL_OPTIONS, {"tooltip": "Incoming options bag. Filled fields are upserted as keywords, or as section overrides when override is on."}),
            },
        }

    RETURN_TYPES = OPTIONS_RETURN_TYPES
    RETURN_NAMES = OPTIONS_RETURN_NAMES
    FUNCTION = "build"
    CATEGORY = "QwenVL-God"

    def build(
        self,
        visual,
        integrated_multimodal_description,
        detailed_description,
        overall_soundscape,
        non_diegetic_music,
        subject_definitions,
        summary,
        retention_analysis,
        placement,
        override=False,
        options=None,
    ):
        fields = {
            "visual": visual,
            "integrated_multimodal_description": integrated_multimodal_description,
            "detailed_description": detailed_description,
            "overall_soundscape": overall_soundscape,
            "non_diegetic_music": non_diegetic_music,
            "subject_definitions": subject_definitions,
            "summary": summary,
            "retention_analysis": retention_analysis,
        }
        filled = {
            key: str(value).strip()
            for key, value in fields.items()
            if str(value or "").strip()
        }
        own = {}
        notes = [f"mode={'override' if override else 'keywords'}"]
        if override:
            if filled:
                own[H3_OVERRIDE_KEY] = filled
            else:
                notes.append("no section fields filled")
        else:
            own[H3_KEYWORD_PLACEMENT_KEY] = placement
            notes.append(f"placement={placement}")
            for key, value in filled.items():
                own[H3_KEYWORD_PREFIX + key] = value
            if not filled:
                notes.append("no section fields filled")
        return options_result(options, own, "QwenVL-God Options (MiniMax Keywords)", notes)


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_MiniMax_Keywords": AILab_QwenVL_MiniMax_Keywords,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_MiniMax_Keywords": "QwenVL-God Options (MiniMax Keywords)",
}
