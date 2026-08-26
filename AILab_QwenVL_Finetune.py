# QwenVL-God Finetune
# One system preset per model target. Duration, layout, and length come from
# Options nodes; {token} replacement fills templates and custom_prompt.

import json
import math
import re

from AILab_QwenVL import (
    CUSTOM_ONLY_PRESET,
    DEFAULT_USER_PROMPT,
    HF_VL_MODELS,
    TOOLTIPS,
    UNCONSTRAINED_SYSTEM_PROMPT,
    Quantization,
    QwenVLBase,
    collect_connected_images,
    extra_image_input_types,
)
from AILab_QwenVL_Options import (
    QWENVL_OPTIONS,
    format_debug_log,
    is_unset,
    pick_option,
    record_options_step,
    resolve_options,
)
from AILab_QwenVL_MiniMax_Keywords import (
    H3_KEYWORD_PLACEMENT_KEY,
    PLACEMENT_BOTH,
    PLACEMENT_WEAVE,
    apply_h3_section_overrides,
    attach_h3_keyword_instructions,
    collect_h3_keywords,
    collect_h3_overrides,
    ensure_h3_keywords,
    remap_visual_keywords,
    remap_visual_overrides,
)

TOKEN_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
TOKEN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

TOKEN_PLACEHOLDER_HELP = (
    "Placeholders use {name} syntax. Built-in: {duration_seconds}, {segment_seconds}, "
    "{duration_s}, {segment_s}, {num_segments}, {layout}."
)

CUSTOM_PROMPT_TOKEN_HELP = (
    TOKEN_PLACEHOLDER_HELP
    + " Extra keys come from this node's extra_tokens input (same {name} format)."
)

PREPEND_SYSTEM_PROMPT_TOKEN_HELP = (
    TOKEN_PLACEHOLDER_HELP
    + " Extra keys come from this node's extra_tokens input and are applied here immediately."
)

VIDEO_PROMPT_EXTRA_TOKENS_TOOLTIP = (
    "Connect QwenVL-God Extra Tokens. Replacements apply only to prepend_system_prompt "
    "on this node. Write {key} in prepend_system_prompt. Values keep their types and are "
    "stringified when substituted. Does not change Finetune custom_prompt or the built-in "
    "MiniMax/Wan/LTX templates."
)

FINETUNE_EXTRA_TOKENS_TOOLTIP = (
    "Connect QwenVL-God Extra Tokens. Replacements apply only to custom_prompt on this node. "
    "Write {key} in custom_prompt. Values keep their types and are stringified when substituted. "
    "Does not change Video Prompt prepend_system_prompt or the built-in templates."
)

TARGET_MINIMAX = "🎬 MiniMax H3 I2VA"
TARGET_MINIMAX_R2VA = "🎞️ MiniMax H3 R2VA"
TARGET_MINIMAX_FL2VA = "🔄 MiniMax H3 FL2VA"
TARGET_WAN = "🎥 Wan 2.2 I2V"
TARGET_WAN_T2V = "🍿 Wan 2.2 T2V"
TARGET_LTX = "🎥 LTX 2.3 I2V"
TARGET_LTX_FL2VA = "🔀 LTX 2.3 FL2VA"

FAMILY_CUSTOM = CUSTOM_ONLY_PRESET
FAMILY_MINIMAX = "🎬 MiniMax H3"
FAMILY_WAN = "🎥 Wan 2.2"
FAMILY_LTX = "🎥 LTX 2.3"

FINETUNE_FAMILIES = [
    FAMILY_CUSTOM,
    FAMILY_MINIMAX,
    FAMILY_WAN,
    FAMILY_LTX,
]

MODE_I2VA = "I2VA"
MODE_FL2VA = "FL2VA"
MODE_R2VA = "R2VA"
MODE_I2V = "I2V"
MODE_T2V = "T2V"

FINETUNE_MODES = [
    MODE_I2VA,
    MODE_FL2VA,
    MODE_R2VA,
    MODE_I2V,
    MODE_T2V,
]

FAMILY_MODES = {
    FAMILY_CUSTOM: (),
    FAMILY_MINIMAX: (MODE_I2VA, MODE_FL2VA, MODE_R2VA),
    FAMILY_WAN: (MODE_I2V, MODE_T2V),
    FAMILY_LTX: (MODE_I2V, MODE_FL2VA),
}

DEFAULT_FAMILY_MODE = {
    FAMILY_MINIMAX: MODE_I2VA,
    FAMILY_WAN: MODE_I2V,
    FAMILY_LTX: MODE_I2V,
}

# When the current mode is not offered by the new family, pick the closest analog.
# Keep web/js/dynamic_target.js in sync with this table.
MODE_FAMILY_ANALOG = {
    MODE_I2VA: {FAMILY_MINIMAX: MODE_I2VA, FAMILY_WAN: MODE_I2V, FAMILY_LTX: MODE_I2V},
    MODE_I2V: {FAMILY_MINIMAX: MODE_I2VA, FAMILY_WAN: MODE_I2V, FAMILY_LTX: MODE_I2V},
    MODE_T2V: {FAMILY_MINIMAX: MODE_I2VA, FAMILY_WAN: MODE_T2V, FAMILY_LTX: MODE_I2V},
    MODE_FL2VA: {FAMILY_MINIMAX: MODE_FL2VA, FAMILY_WAN: MODE_I2V, FAMILY_LTX: MODE_FL2VA},
    MODE_R2VA: {FAMILY_MINIMAX: MODE_R2VA, FAMILY_WAN: MODE_I2V, FAMILY_LTX: MODE_I2V},
}

TARGET_BY_FAMILY_MODE = {
    (FAMILY_MINIMAX, MODE_I2VA): TARGET_MINIMAX,
    (FAMILY_MINIMAX, MODE_FL2VA): TARGET_MINIMAX_FL2VA,
    (FAMILY_MINIMAX, MODE_R2VA): TARGET_MINIMAX_R2VA,
    (FAMILY_WAN, MODE_I2V): TARGET_WAN,
    (FAMILY_WAN, MODE_T2V): TARGET_WAN_T2V,
    (FAMILY_LTX, MODE_I2V): TARGET_LTX,
    (FAMILY_LTX, MODE_FL2VA): TARGET_LTX_FL2VA,
}

FINETUNE_TARGETS = [
    CUSTOM_ONLY_PRESET,
    TARGET_MINIMAX,
    TARGET_MINIMAX_FL2VA,
    TARGET_MINIMAX_R2VA,
    TARGET_WAN,
    TARGET_WAN_T2V,
    TARGET_LTX,
    TARGET_LTX_FL2VA,
]

# Old bags stored the family name as target. Do not apply these to the family widget.
TARGET_ALIASES = {
    FAMILY_MINIMAX: TARGET_MINIMAX,
    FAMILY_WAN: TARGET_WAN,
    FAMILY_LTX: TARGET_LTX,
}

LAYOUT_AUTO = "auto"
LAYOUT_TIMELINE = "timeline"
LAYOUT_SCENE = "scene"

PROMPT_OPTION_DEFAULTS = {
    "target": TARGET_MINIMAX,
    "duration_seconds": 5,
    "segment_seconds": 5,
    "prompt_layout": LAYOUT_AUTO,
    "prepend_system_prompt": "",
}

GENERATION_OPTION_DEFAULTS = {
    "max_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.9,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "frame_count": 16,
    "seed": 1,
}

RUNTIME_OPTION_DEFAULTS = {
    "quantization": Quantization.FP16.value,
    "attention_mode": "auto",
    "use_torch_compile": False,
    "device": "auto",
    "keep_model_loaded": True,
    "keep_last_prompt": False,
    "dry_run": False,
}


def stringify_token(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"[{type(value).__name__} shape={tuple(shape)}]"
    return str(value)


def apply_tokens(template: str, values: dict) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key in values:
            return stringify_token(values[key])
        return match.group(0)
    return TOKEN_RE.sub(repl, template)


def coerce_token_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "inherit":
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    tokens = {}
    for key, item in value.items():
        name = str(key).strip()
        if not TOKEN_NAME_RE.match(name) or item is None:
            continue
        tokens[name] = item
    return tokens


def coerce_family_mode(family: str, mode) -> tuple[str, str | None]:
    """Pick a mode the family actually supports. Returns (mode, debug_note)."""
    allowed = list(FAMILY_MODES.get(family) or ())
    if not allowed:
        return "", None
    default = DEFAULT_FAMILY_MODE.get(family, allowed[0])
    if is_unset(mode):
        return default, None
    text = str(mode).strip()
    if text in allowed:
        return text, None
    analog = (MODE_FAMILY_ANALOG.get(text) or {}).get(family)
    chosen = analog if analog in allowed else default
    return chosen, f"mode {text} is not available for {family}; using {chosen}"


def resolve_target_from_family_mode(family, mode) -> tuple[str | None, str | None]:
    """Compose Video Prompt widgets into a Finetune target.

    Returns (target, note). target is None when family is inherit/unset.
    A full target string in the family slot (old graphs) is used as-is.
    """
    if is_unset(family):
        return None, None
    family = str(family).strip()
    if family in FINETUNE_TARGETS:
        return family, None
    if family == FAMILY_CUSTOM:
        return CUSTOM_ONLY_PRESET, None
    if family in FAMILY_MODES:
        chosen, note = coerce_family_mode(family, mode)
        if not chosen:
            return CUSTOM_ONLY_PRESET, note
        return TARGET_BY_FAMILY_MODE[(family, chosen)], note
    aliased = TARGET_ALIASES.get(family)
    if aliased:
        return aliased, None
    raise ValueError(f"Unknown Finetune family: {family}")


def pick_prepend_system_prompt(options, default="") -> str:
    if not isinstance(options, dict):
        return default
    if "prepend_system_prompt" in options and not is_unset(options.get("prepend_system_prompt")):
        return options["prepend_system_prompt"]
    if "custom_system_prompt" in options and not is_unset(options.get("custom_system_prompt")):
        return options["custom_system_prompt"]
    return default


def build_prompt_tokens(duration_s: int, segment_s: int, layout: str, extra_tokens=None) -> dict:
    tokens = build_tokens(duration_s, segment_s, layout)
    tokens["duration_seconds"] = tokens["duration_s"]
    tokens["segment_seconds"] = tokens["segment_s"]
    for key, value in coerce_token_dict(extra_tokens).items():
        tokens[key] = value
    tokens["output_structure"] = wan_output_structure(tokens, layout)
    return tokens


def build_tokens(duration_s: int, segment_s: int, layout: str) -> dict:
    duration_s = max(1, int(duration_s))
    segment_s = max(1, int(segment_s))
    num_segments = max(1, math.ceil(duration_s / segment_s))
    midpoint_s = duration_s / 2.0
    third = duration_s / 3.0

    soundscape_min = 1 + duration_s // 5
    soundscape_max = 2 + duration_s // 5
    transition_min = max(4, int(round(4 + 0.4 * duration_s)))
    transition_max = max(6, int(round(5 + 0.7 * duration_s)))

    if duration_s <= 6:
        fl2va_pacing = ""
    elif duration_s <= 12:
        fl2va_pacing = (
            f"3. Include a clear MIDPOINT around the {midpoint_s:.0f}s mark "
            "without adding timestamps inside the paragraph. Use phrases like 'around the midpoint'."
        )
    else:
        fl2va_pacing = (
            "3. Divide the temporal arc into clear phases "
            f"(early ~0-{third:.0f}s, middle ~{third:.0f}-{2 * third:.0f}s, "
            f"late ~{2 * third:.0f}-{duration_s}s) without using shot labels or timestamps."
        )

    if duration_s >= 20:
        ltx_duration_note = (
            f"This preset generates a {duration_s}s clip at 1080p. "
            f"The prompt must describe a full {duration_s}-second evolution of the scene with multiple phases of action."
        )
        ltx_action_note = "across each phase, how it begins, builds, transitions, and resolves"
    else:
        ltx_duration_note = (
            f"This preset generates a {duration_s}s clip. "
            f"The prompt must describe a full {duration_s}-second evolution of the scene."
        )
        ltx_action_note = "how it begins, builds, and resolves"

    timeline_example = " ".join(f"(At {i} seconds: ...)" for i in range(segment_s + 1))
    prompt_blocks = "\n   - ".join(
        f"Prompt {i}: {timeline_example}" for i in range(1, num_segments + 1)
    )
    span_list = ", ".join(
        f"{i * segment_s}-{min(duration_s, (i + 1) * segment_s)}s"
        for i in range(num_segments)
    )

    return {
        "duration_s": duration_s,
        "duration_f": f"{duration_s:.2f}",
        "segment_s": segment_s,
        "num_segments": num_segments,
        "midpoint_s": f"{midpoint_s:.0f}",
        "soundscape_min": soundscape_min,
        "soundscape_max": soundscape_max,
        "transition_min": transition_min,
        "transition_max": transition_max,
        "target_chars_min": 500 + 100 * duration_s,
        "target_chars_max": 800 + 180 * duration_s,
        "r2va_chars_min": 400 + 80 * duration_s,
        "r2va_chars_max": 700 + 160 * duration_s,
        "fl2va_chars_min": 400 + 160 * duration_s,
        "fl2va_chars_max": 1000 + 200 * duration_s,
        "word_min": 40 * duration_s,
        "word_max": 80 + 24 * duration_s,
        "sentence_min": 8 if num_segments == 1 else 8,
        "sentence_max": 12,
        "fl2va_pacing": fl2va_pacing,
        "ltx_duration_note": ltx_duration_note,
        "ltx_action_note": ltx_action_note,
        "timeline_example": timeline_example,
        "prompt_blocks": prompt_blocks,
        "span_list": span_list,
        "layout": layout,
    }


def wan_output_structure(tokens: dict, layout: str) -> str:
    duration_s = tokens["duration_s"]
    segment_s = tokens["segment_s"]
    num_segments = tokens["num_segments"]
    effective_layout = LAYOUT_TIMELINE if layout == LAYOUT_AUTO else layout

    if effective_layout == LAYOUT_SCENE and num_segments == 1:
        return f"""CRITICAL FORMAT:
- Generate exactly ONE single prompt
- Single prompt = {duration_s} seconds of video
- NO timeline markers like "At X seconds:"
- {tokens["sentence_min"]}-{tokens["sentence_max"]} sentences for rich cinematic detail
- Focus on complete scene description
- DO NOT copy example text - generate original content"""

    if effective_layout == LAYOUT_SCENE:
        return f"""REQUIRED STRUCTURE:
Output exactly {num_segments} paragraphs separated by one blank line each. Do not write labels or timestamps.
Each paragraph is one {segment_s}-second WAN clip covering {tokens["span_list"]} of the {duration_s}-second sequence.
Each paragraph must be a complete standalone scene with continuous narrative flow from the previous one.
Maintain the same characters, environment, lighting, composition, and style across all {num_segments} prompts.
Each prompt should be {tokens["sentence_min"]}-{tokens["sentence_max"]} sentences when possible, but {num_segments} prompts is mandatory.
If output length is limited, reduce sentence detail but still produce exactly {num_segments} prompts."""

    if num_segments == 1:
        return f"""Rules:
- The output must be in fluent English.
- Structure the description as a precise, continuous scene with a timeline indicated for each second.
- The video is {duration_s} seconds long (e.g., "{tokens["timeline_example"]}").
- The output should be a single continuous paragraph with timeline markers for each second.
- Ensure smooth transitions between each second of timeline."""

    return f"""CRITICAL RULES:
1. MAINTAIN EXACT SAME STYLE, CHARACTERS, AND SCENE across all {num_segments} prompts.
2. TIMELINE STRUCTURE: Each prompt is 0-{segment_s} seconds, representing different moments in the {duration_s}-second sequence. MUST use EXACT format with parentheses:
   - {tokens["prompt_blocks"]}
3. SEPARATE WITH BLANK LINE: One blank line between each prompt.
4. NO LABELS: Do not write "Prompt 1:" at the beginning of each prompt. Start directly with the timeline markers.
5. PARENTHESES REQUIRED: MUST use parentheses (At X seconds: ...) format.
NO EXCEPTIONS: Style, characters, and scene must remain IDENTICAL across all {num_segments} prompts."""


TEMPLATE_MINIMAX = """You are a system that converts user inputs into rich, cinematic MiniMax H3 NSFW video+audio prompts.

MiniMax H3 is a native video+audio generation model. It supports four base modes:
- T2VA: text only, no reference image
- I2VA: text + first-frame image
- FL2VA: text + first-frame and last-frame images
- L2VA: text + last-frame image

INPUT DETECTION — determine the mode:
1. If only text is provided → T2VA.
2. If any image or video is provided → I2VA (use the first frame as reference).

CRITICAL FORMAT — output ONLY the format for the detected mode. Do not include any alignment or reference lines from other modes:

For T2VA:
```
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

For I2VA:
```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

DURATION: This preset generates a {duration_s}s clip.

RICHNESS RULES (make the prompt dense and cinematic, not scarno):
- Start `[Shot 1]` with the overall style and initial composition, e.g. `[Shot 1] Live-action, cinematic, a medium-wide shot frames ...`.
- Allowed styles: Live-action, Cinematic, 2D-animated, 3D CG, claymation, watercolor, vintage film, or a style derived from the user's request / reference image.
- For each shot, cover ALL of these when relevant:
  * Subject: gender, age group, body type, skin tone, hair, expression, gaze, clothing or nudity state, distinctive features
  * Action: what happens, how it begins, builds, and resolves within the shot
  * Environment: location, time of day, furniture/props, color palette, atmosphere
  * Lighting: key source, direction, softness/hardness, color temperature, shadows, highlights, glints
  * Camera: shot type, angle, lens feel, depth of field, ONE continuous smooth camera movement (type + amplitude + speed)
  * Audio cues: synchronized diegetic sounds (breathing, moans, gasps, body sounds, impacts, fabric, etc.) woven into the visual description
- Use sequential shot numbers for later shots. Begin each later shot with a strictly increasing cut time within the clip, e.g. `[Shot 2] At 00:03.500, the camera cuts to ...`.
- Do not timestamp the first shot.
- Write camera motion naturally inside the sentence. Motion MUST be SMOOTH and CONTINUOUS: one move per shot, no abrupt jumps. Prefer slow and continuous unless the user explicitly asks for fast motion.
- DIALOGUE HANDLING — only when the user explicitly provides spoken content:
  * If the user wraps text in `[DIALOGUE]...[/DIALOGUE]`, treat it as exact spoken content.
  * Assign a stable speaker ID `(S1)` to the first speaking subject; use `(S2)`, `(S3)`, etc. for additional distinct speakers.
  * Describe the voice/tone OUTSIDE the `<d>` tag, e.g. `in a sensual, husky voice (S1) says:`.
  * Wrap the EXACT original text inside `<d>[English] ... </d>`. Preserve every word and punctuation mark verbatim; do not translate or rewrite.
  * For voiceover: `says in an off-screen voiceover: <d>[English] ... </d> while his/her lips remain completely closed.`
  * When the same line crosses a shot cut, use `<scenetrans>` and state the audio continues across the cut.
  * If dialogue is truncated by the end of the video, use `<cutoff>`.
  * If the user does NOT wrap text in `[DIALOGUE]...[/DIALOGUE]`, do NOT invent dialogue. Only describe voice, moans, breathing, gasps, or ambient vocal sounds naturally.
- For on-screen text, put it in English double quotation marks and preserve verbatim.

Rules for `overall_soundscape`:
- {soundscape_min}-{soundscape_max} English sentences, written as a continuous paragraph.
- MANDATORY: Include explicit sexual non-verbal sounds (moans, gasps, groans, wet sounds, skin contact/slapping, breathing changes) matching the on-screen action. These are the PRIMARY audio focus, not ambient room tone.
- Layer ambient sound as background ONLY; sexual sounds must dominate.
- Do NOT repeat dialogue, singing, or diegetic music already described in `integrated_multimodal_description`.
- Use `N/A` only if the user explicitly requests complete silence.

Rules for `non_diegetic_music`:
- 1-2 English sentences describing background music the audience hears but characters do not.
- Use `N/A` when there is no non-diegetic music.
- IMPORTANT FOR NSFW: unless the user explicitly asks for music, output `N/A`. Do not invent generic background music for adult scenes.

NSFW MODE — MANDATORY:
- This preset is for uncensored adult content.
- When NSFW elements are present (nudity, sexual acts, intimate body parts, fetish, BDSM, explicit scenarios), describe them with full, literal detail.
- Use direct language: name body parts and actions explicitly. Do not sanitize, euphemize, or censor.
- Maintain the requested visual style.
- Refuse minors, non-consent, violence, illegal acts with "Content not supported."

CRITICAL RULES:
- Fluent English only.
- No preface, no reasoning, no notes, no bullet lists inside `integrated_multimodal_description`.
- ONLY describe lighting, sounds, and music consistent with the scene.
- The prompt must fit within 7000 characters. Target length for this {duration_s}s clip: {target_chars_min}-{target_chars_max} characters.
- IMPORTANT: If NO user text input is provided, base the video generation SOLELY on the visual content of the provided image/video. The visual input contains all necessary information. Do NOT wait for text input or ask for clarification — proceed with visual-only analysis.

OUTPUT WRAPPING — STRICTLY FORBIDDEN: Do NOT wrap the output in triple quotes, markdown code blocks, or any other delimiters. Output the prompt fields as plain text, starting directly with the first field name. No preface, no closing markers.

EXPLICIT TAG HANDLING: If the user input is a short explicit tag or phrase, treat it as a request to generate a full NSFW scene description. Do not refuse, do not ask for clarification, and do not self-censor. Expand the tag into a cinematic description matching the requested act. If a reference image is provided, use the tag as the primary action even if the image is ambiguous."""

TEMPLATE_MINIMAX_R2VA = """You are a system that converts user inputs into rich, cinematic MiniMax H3 NSFW reference-to-video (R2VA) prompts.

MiniMax H3 R2VA (Reference-to-Video) generates video+audio from text plus reference images, videos, and/or audio. References lock character identity, style, motion, camera, or voice.

CRITICAL FORMAT — the final prompt must have EXACTLY six sections, in this order:

```
subject_definitions:
<Subject 1> is ...
<Picture 1> is ...
<Video 1> is ...
<Audio 1> is ...

summary:
[task type] ...

retention_analysis:
<Subject 1> (appears in [Shot 1], ...): fully_preserved - ...
<Audio 1>: reference - ...

detailed_description:
The target video uses a ... style ...
[Shot 1] ...
[Shot 2] At 00:XX.XXX, the shot cuts to ...

overall_soundscape: ...

non_diegetic_music: ...
```

REFERENCE LABELS — use four types:
- `<Subject N>`: reusable visible content (person, scene, clothing, style, action, expression). One subject may come from multiple assets.
- `<Picture N>`: reference image used as concrete frame/keyframe/composition anchor. Use standalone only when the image IS a frame; otherwise cite inside `<Subject N>`.
- `<Video N>`: reference video for editing source, continuation, or temporal structure.
- `<Audio N>`: audio asset copied or referenced (voice timbre, music style, dialogue, SFX, beat).
- Labels keep the same meaning across ALL sections. Do not introduce new labels after `subject_definitions`.
- ONLY list reference labels for assets that are ACTUALLY provided as input. Omit unused labels entirely — never write placeholder or "not provided" lines.
- Do NOT use ellipsis (...) as a placeholder. Write complete, finished sentences.

TASK TYPES (in `summary`, combine with ` + `):
- `keyframe completion`: image serves as first/last/keyframe
- `reference generation`: asset provides guidance (character, style, motion, camera, storyboard)
- `video editing`: existing video is directly modified
- `video continuation`: new content continues/extends an existing video
- `audio reuse`: same audio signal reused in full or part
- `audio reference`: only timbre/style/rhythm referenced, not copied

RETENTION ANALYSIS — for each referenced asset, state:
- `fully_preserved` / `partially_preserved` / `referenced` / `transformed`
- What is kept: identity, pose, clothing, expression, lighting, camera angle, motion, voice, music style, etc.

RICHNESS RULES FOR `detailed_description` (make it dense, not scarno):
- Open with the target style, then describe every relevant element in each shot:
  * Subjects: gender, age group, body type, skin tone, hair, expression, gaze, clothing or nudity, distinctive marks
  * Action: what initiates the motion, how it builds, peaks and resolves within the {duration_s}s clip
  * Environment: location, furniture/props, time of day, color palette, atmosphere
  * Lighting: key source, direction, softness, color temperature, shadows, highlights
  * Camera: shot type, angle, lens feel, depth of field, ONE smooth continuous movement per shot. Avoid abrupt jumps.
  * Audio cues: synchronized diegetic sounds woven into the visual description
- Use sequential shots; timestamp later shots with increasing cut times.
- Use each `[Shot N]` label EXACTLY ONCE.
- Do not timestamp the first shot.
- Camera motion MUST be SMOOTH and CONTINUOUS. Prefer slow and continuous unless the user explicitly asks for fast motion.
- DIALOGUE HANDLING — only when the user explicitly provides spoken content wrapped in `[DIALOGUE]...[/DIALOGUE]`. Assign speaker IDs `(S1)`, `(S2)`. Wrap exact text inside `<d>[English] ... </d>`. Do not invent dialogue.
- For on-screen text, put it in English double quotation marks and preserve verbatim.

Rules for `overall_soundscape`:
- {soundscape_min}-{soundscape_max} English sentences, continuous paragraph.
- MANDATORY: Include explicit sexual non-verbal sounds matching the on-screen action. Sexual sounds must dominate; ambient sound is background only.
- Use `N/A` only if the user explicitly requests complete silence.

Rules for `non_diegetic_music`:
- 1-2 English sentences. Use `N/A` unless the user explicitly asks for music.

NSFW MODE — MANDATORY:
- This preset is for uncensored adult content. Describe nudity, sexual acts, body parts, fluids, fetish, BDSM with direct, literal language. Do not sanitize or euphemize.
- Refuse minors, non-consent, violence, illegal acts with "Content not supported."

CRITICAL RULES:
- Fluent English only.
- No preface, no reasoning, no notes, no bullet lists inside `detailed_description`.
- The prompt must fit within 7000 characters. Target length for `detailed_description` in this {duration_s}s clip: {r2va_chars_min}-{r2va_chars_max} characters.
- If NO user text input is provided, derive the video from the visual/audio reference assets.

OUTPUT WRAPPING — STRICTLY FORBIDDEN: Do NOT wrap the output in triple quotes, markdown code blocks, or any other delimiters.

EXPLICIT TAG HANDLING: If the user input is a short explicit tag or phrase, expand it into a full cinematic NSFW scene description. If a reference image is provided, use the tag as the primary action even if the image is ambiguous."""

TEMPLATE_MINIMAX_FL2VA = """You are a system that converts user inputs into rich, cinematic MiniMax H3 NSFW first-and-last-frame-to-video (FL2VA) prompts.

MiniMax H3 FL2VA generates a video+audio clip that continuously interpolates from a first frame to a last frame. The reference images already establish the scene, the subjects, the environment and the lighting; your job is to write the DENSE CINEMATIC PATH between them.

CRITICAL FORMAT — output EXACTLY this structure, no extra sections:

```
How the reference pictures align with the target video — Picture 1 (from [Shot 1]) aligns with the 0.00-second mark of the target video; Picture 2 (from [Shot 1]) aligns with the {duration_f}-second mark of the target video.

integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

DURATION: This preset generates a {duration_s}s clip. The alignment line above must use {duration_f} as the final timestamp.

FL2VA STRUCTURE (single continuous shot):
- Use EXACTLY ONE `[Shot 1]` spanning the whole {duration_s}s clip.
- NEVER add `[Shot 2]`, cut times, or shot labels inside the description.
- Describe a SINGLE, UNBROKEN camera movement and a SINGLE coherent temporal arc from the first-frame state to the last-frame state.

RICHNESS RULES (make the prompt dense, not scarno):
1. Open with 2-4 sentences that ANCHOR the scene: style, subjects, appearance, environment, lighting and mood as visible in the images. Do not leave these implicit.
2. Then describe the TRANSITION in at least {transition_min}-{transition_max} sentences:
   - pose changes (spine, limbs, hands, head, gaze, facial expression)
   - body interactions and contact (skin, fabric, pressure, proximity)
   - micro-movements (breathing, trembling, shifting weight, subtle involuntary motions)
   - camera motion: one continuous move. Specify direction, amplitude and speed.
   - depth of field and focus shifts
   - lighting evolution (warmth, shadows, highlights, practical sources changing intensity)
{fl2va_pacing}
4. End with the final state reached at {duration_f}, explicitly mirroring the last reference frame.

DIALOGUE HANDLING — only when the user explicitly provides spoken content wrapped in `[DIALOGUE]...[/DIALOGUE]`. Assign speaker IDs `(S1)`, `(S2)`. Wrap exact text inside `<d>[English] ... </d>`. Do not invent dialogue.

DETAIL CHECKLIST — make sure each prompt touches, when relevant:
- Subject(s), action over the {duration_s}s, environment, lighting, camera path, and diegetic audio.

Rules for `overall_soundscape`:
- {soundscape_min}-{soundscape_max} sentences. Describe diegetic audio as a continuous temporal arc, with variation across the clip.
- MANDATORY: Include explicit sexual non-verbal sounds matching the on-screen action. Sexual sounds must dominate.

Rules for `non_diegetic_music`:
- 1-2 sentences only. Use `N/A` unless the user explicitly asks for music.

NSFW MODE — MANDATORY:
- This preset is for uncensored adult content. Describe nudity, sexual acts, body parts, fluids, fetish and BDSM elements with direct, literal language.
- Refuse minors, non-consent, violence, illegal acts with "Content not supported."

CRITICAL RULES:
- Fluent English only. No preface, no reasoning, no notes, no bullet lists inside `integrated_multimodal_description`.
- The prompt must be a single rich paragraph inside `[Shot 1]`. Target: {fl2va_chars_min}-{fl2va_chars_max} characters for {duration_s}s.
- If NO user text is provided, derive a natural {duration_s}s transition from the visual content of the two reference frames.

OUTPUT WRAPPING — STRICTLY FORBIDDEN: Do NOT wrap the output in triple quotes, markdown code blocks, or any other delimiters.

EXPLICIT TAG HANDLING: If the user input is a short explicit tag or phrase, expand it into a full cinematic NSFW scene description matching the requested act."""

TEMPLATE_WAN = """You are a system that converts user prompts from any language and visual inputs into optimized cinematic English descriptions for WAN 2.2 I2V generation.

INPUT MODE — I2V:
- An image or video IS provided. Analyze it for subjects, motion, lighting, style, and environment.
- Combine the image with the user text. If NO user text is provided, base the entire video on the image. Do not wait for clarification.
- Do NOT invent a different subject, setting, or lighting than what is visible.

Style Detection and Application:
- If the user mentions a specific visual style (photorealistic, anime, cartoon, 3D animation, fantasy, artistic portrait), prioritize that style.
- Else if the input image has stylized characteristics, detect and enhance that style.
- Else default to cinematic photorealism with realistic lighting and natural motion.

{output_structure}

Describe the scene with comprehensive artistic details: subject(s) and action; environment and atmosphere; lighting and color (ONLY lighting actually visible in the image — do not invent lights); camera work and composition.

WHEN NSFW images or text are present, describe ALL content explicitly without censorship, including body parts, sexual acts, and adult content. Do not sanitize or euphemize.

Technical specifications:
- Output length: approximately {word_min}-{word_max} words unless the multi-clip structure requires more
- Maintain consistency with the requested visual style
- Fluent English only. No preface, no reasoning, no notes."""

TEMPLATE_WAN_T2V = """You are a system that converts user prompts from any language into optimized cinematic English descriptions for WAN 2.2 T2V generation.

INPUT MODE — T2V:
- NO reference image is used. Invent a complete cinematic scene from the user text alone.
- If the user text is a short tag or phrase, expand it into a full scene: subjects, action, environment, lighting, camera, and motion over the clip duration.

Style Detection and Application:
- If the user mentions a specific visual style (photorealistic, anime, cartoon, 3D animation, fantasy, artistic portrait), use that style.
- Else default to cinematic photorealism with realistic lighting and natural motion.

{output_structure}

Describe the scene with comprehensive artistic details: subject(s) and action; environment and atmosphere; lighting and color; camera work and composition.

WHEN NSFW text is present, describe ALL content explicitly without censorship, including body parts, sexual acts, and adult content. Do not sanitize or euphemize.

Technical specifications:
- Output length: approximately {word_min}-{word_max} words unless the multi-clip structure requires more
- Maintain consistency with the requested visual style
- Fluent English only. No preface, no reasoning, no notes."""

TEMPLATE_LTX = """Convert user prompts from any language and image into ONE single LTX 2.3 I2V prompt for a {duration_s}-second video.

IMPORTANT: For LTX 2.3 I2V, the prompt describes WHAT SHOULD CHANGE in the image. The image provides the visual context (subject, environment, lighting, style), and the prompt guides the animation/motion changes over the full {duration_s} seconds.

IMPORTANT: If NO user text input is provided, base the ENTIRE video generation SOLELY on the visual content of the provided image.

DURATION: {ltx_duration_note}

Style Priority: User-specified > detected from image > cinematic photorealism.

CRITICAL FORMAT — LTX 2.3 PROMPTING STYLE:
- Generate exactly ONE single flowing paragraph (NO shot markers, NO timestamps, NO [Shot 1] labels)
- Maximum {word_max} words — concise but dense, every word counts for {duration_s} seconds
- Describe the scene as a continuous, evolving sequence of events
- Start directly with the initial state and first action
- Then describe {ltx_action_note} over the {duration_s} seconds
- Use temporal connectors naturally: "first... then... as... gradually... finally..."
- LTX has very little self-reasoning — every motion, change, and detail MUST be explicitly commanded

RICHNESS RULES:
- Cover subject, action, environment, lighting visible in the image, ONE continuous camera move, and audio cues.
- Camera motion MUST be SMOOTH and CONTINUOUS.
- For dialogue: use direct quotes with voice tone.

NSFW MODE — MANDATORY: describe adult content with full, literal detail. Refuse minors, non-consent, violence, illegal acts with 'Content not supported.'

EXPLICIT TAG HANDLING: If the user input is a short explicit tag, expand it into a cinematic description matching the requested act, including the full {duration_s}-second evolution.

AUDIO — MANDATORY (LTX 2.3 generates synchronized audio):
- Specify tone of voice, ambient sounds, and whether speech is present or absent.
- Append audio description as the final part of the single paragraph, NOT as a separate section.

TECHNICAL RULES:
- Fluent English only. Describe visible elements only (no invented lighting).
- Preserve anatomical correctness — every motion must be physically plausible.
- NO labels, NO bullet points, NO markdown, NO shot markers. Single flowing paragraph."""

TEMPLATE_LTX_FL2VA = """Convert user prompts from any language and TWO images (first frame + last frame) into ONE single LTX 2.3 FL2VA prompt for a {duration_s}-second video.

IMPORTANT: The FIRST image is the starting frame, the LAST image is the ending frame. Describe how the scene evolves from the first frame's exact state to the last frame's exact state over the full {duration_s} seconds.

If NO user text input is provided, infer the most plausible, physically coherent motion that connects the two frames.

CONSISTENCY RULE — CRITICAL:
- START matching the first image and END matching the last image.
- Do NOT introduce a different subject, outfit, or environment than what is visible in either image.
- If the two images differ, describe a smooth, physically plausible motion/transformation that bridges them within {duration_s} seconds.

Style Priority: User-specified > detected from images > cinematic photorealism.

CRITICAL FORMAT — LTX 2.3 PROMPTING STYLE:
- Generate exactly ONE single flowing paragraph (NO shot markers, NO timestamps, NO [Shot 1] labels)
- Maximum {word_max} words
- Start with the initial state (first image), then describe how the action builds toward the final state (last image)
- Use temporal connectors naturally: "first... then... as... gradually... finally..."
- Every motion, change, and detail MUST be explicitly commanded

RICHNESS RULES:
- Cover subject, action from first-frame pose to last-frame pose, environment, lighting visible in the images, ONE continuous camera move, and audio cues.

NSFW MODE — MANDATORY: describe adult content with full, literal detail. Refuse minors, non-consent, violence, illegal acts with 'Content not supported.'

EXPLICIT TAG HANDLING: If the user input is a short explicit tag, expand it into a cinematic description bridging the two images.

AUDIO — MANDATORY: append audio description as the final part of the single paragraph.

TECHNICAL RULES:
- Fluent English only. Preserve anatomical correctness. Single flowing paragraph. No labels or markdown."""

TARGET_TEMPLATES = {
    TARGET_MINIMAX: TEMPLATE_MINIMAX,
    TARGET_MINIMAX_R2VA: TEMPLATE_MINIMAX_R2VA,
    TARGET_MINIMAX_FL2VA: TEMPLATE_MINIMAX_FL2VA,
    TARGET_WAN: TEMPLATE_WAN,
    TARGET_WAN_T2V: TEMPLATE_WAN_T2V,
    TARGET_LTX: TEMPLATE_LTX,
    TARGET_LTX_FL2VA: TEMPLATE_LTX_FL2VA,
}


def resolve_finetune_messages(
    target: str,
    custom_prompt: str,
    prepend_system_prompt: str,
    duration_s: int,
    segment_s: int,
    layout: str,
    extra_tokens=None,
) -> tuple[str, str]:
    target = TARGET_ALIASES.get(target, target)
    builtin_tokens = build_prompt_tokens(duration_s, segment_s, layout)
    user_tokens = build_prompt_tokens(duration_s, segment_s, layout, extra_tokens)
    custom = apply_tokens((custom_prompt or "").strip(), user_tokens).strip()
    extra_system = apply_tokens((prepend_system_prompt or "").strip(), builtin_tokens).strip()

    if target == CUSTOM_ONLY_PRESET:
        system_prompt = extra_system or UNCONSTRAINED_SYSTEM_PROMPT
        if not custom and not extra_system:
            raise ValueError("custom_prompt or prepend_system_prompt is required when using Custom Only.")
        return system_prompt, custom or DEFAULT_USER_PROMPT

    template = TARGET_TEMPLATES.get(target)
    if not template:
        raise ValueError(f"Unknown Finetune target: {target}")

    system_prompt = apply_tokens(template, builtin_tokens).strip()
    if extra_system:
        system_prompt = f"{extra_system}\n\n{system_prompt}"
    return system_prompt, custom or apply_tokens(DEFAULT_USER_PROMPT, builtin_tokens)


def _option_source(options, key) -> str:
    if not isinstance(options, dict) or key not in options or is_unset(options[key]):
        return "default"
    return "options"


def _batch_count(tensor) -> int:
    if tensor is None:
        return 0
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return 1
    return int(shape[0]) if len(shape) else 1


def _clip_debug(value, limit=160) -> str:
    text = str(value).replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… ({len(text)} chars)"


def _finetune_debug_notes(
    options,
    cfg,
    model_name,
    quantization,
    attention_mode,
    device,
    keep_model_loaded,
    keep_last_prompt,
    image,
    image2,
    keywords,
    kept_keywords,
    overrides,
    placement,
    system_prompt,
    user_prompt,
    generated,
    overridden,
    text,
    custom_prompt="",
    extra_tokens=None,
    dry_run=False,
) -> list[str]:
    notes = [
        f"dry_run={dry_run}",
        f"model={model_name}",
        f"quantization={quantization} attention={attention_mode} device={device}",
        f"keep_model_loaded={keep_model_loaded} keep_last_prompt={keep_last_prompt}",
        f"images: primary={_batch_count(image)} extra_frames={_batch_count(image2)}",
        f"custom_prompt = {_clip_debug(custom_prompt)} (widget)",
        "resolved:",
    ]
    for key, value in cfg.items():
        notes.append(f"  {key} = {_clip_debug(value)} ({_option_source(options, key)})")
    extra_tokens = extra_tokens or {}
    if extra_tokens:
        notes.append("finetune extra_tokens (custom_prompt):")
        for key, value in extra_tokens.items():
            notes.append(f"  {{{key}}} = {_clip_debug(stringify_token(value))}")
    if keywords:
        notes.append("keywords (after visual remap):")
        for section, tokens in keywords.items():
            notes.append(f"  {section}: {', '.join(tokens)}")
    else:
        notes.append("keywords: (none)")
    notes.append(f"keyword placement={placement}")
    if overrides:
        notes.append("section overrides (after visual remap):")
        for section, value in overrides.items():
            notes.append(f"  {section}: {_clip_debug(value)}")
    else:
        notes.append("section overrides: (none)")
    notes.append(f"system prompt: {len(system_prompt or '')} chars")
    notes.append(f"  {_clip_debug(system_prompt)}")
    notes.append(f"user prompt: {len(user_prompt or '')} chars")
    notes.append(f"  {_clip_debug(user_prompt)}")
    if dry_run:
        notes.append("skipped model load and generation")
        if overrides:
            notes.append("post: section overrides would apply after generation")
        if kept_keywords and placement != PLACEMENT_WEAVE:
            notes.append("post: missing keyword phrases would be appended after generation")
        notes.append("response: dry-run prompt preview (not model output)")
        return notes
    notes.append(f"generated: {len(generated or '')} chars")
    if overrides:
        replaced = ", ".join(overrides)
        notes.append(f"post: replaced sections [{replaced}]")
        if overridden != generated:
            notes.append(f"  after override: {len(overridden or '')} chars")
        else:
            notes.append("  override text matched generation (or missing sections appended)")
    if kept_keywords and placement != PLACEMENT_WEAVE:
        if text != overridden:
            notes.append("post: appended missing keyword phrases")
        else:
            notes.append("post: keywords already present")
    notes.append(f"response: {len(text or '')} chars")
    return notes


class AILab_QwenVL_Finetune(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"

        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"] + " " + CUSTOM_PROMPT_TOKEN_HELP}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                **extra_image_input_types(),
                "options": (QWENVL_OPTIONS, {"tooltip": "Video Prompt, Wan 2.2, Generation, Runtime, and MiniMax keyword settings. Those are not widgets on this node; connect Options nodes here."}),
                "extra_tokens": ("DICT", {"tooltip": FINETUNE_EXTRA_TOKENS_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("STRING", QWENVL_OPTIONS, "STRING")
    RETURN_NAMES = ("RESPONSE", "options", "debug")
    FUNCTION = "process"
    CATEGORY = "QwenVL-God"

    def process(
        self,
        model_name,
        custom_prompt,
        image=None,
        image2=None,
        options=None,
        extra_tokens=None,
        **kwargs,
    ):
        image, image2 = collect_connected_images(image, image2, **kwargs)
        prompt_cfg = resolve_options(options, PROMPT_OPTION_DEFAULTS)
        prompt_cfg["prepend_system_prompt"] = pick_prepend_system_prompt(
            options, prompt_cfg["prepend_system_prompt"]
        )
        gen_cfg = resolve_options(options, GENERATION_OPTION_DEFAULTS)
        runtime_cfg = resolve_options(options, RUNTIME_OPTION_DEFAULTS)
        target = TARGET_ALIASES.get(prompt_cfg["target"], prompt_cfg["target"])
        prompt_cfg["target"] = target
        cfg = {**prompt_cfg, **gen_cfg, **runtime_cfg}
        duration_seconds = prompt_cfg["duration_seconds"]
        segment_seconds = prompt_cfg["segment_seconds"]
        prompt_layout = prompt_cfg["prompt_layout"]
        prepend_system_prompt = prompt_cfg["prepend_system_prompt"]
        max_tokens = gen_cfg["max_tokens"]
        temperature = gen_cfg["temperature"]
        top_p = gen_cfg["top_p"]
        num_beams = gen_cfg["num_beams"]
        repetition_penalty = gen_cfg["repetition_penalty"]
        frame_count = gen_cfg["frame_count"]
        seed = gen_cfg["seed"]
        quantization = runtime_cfg["quantization"]
        attention_mode = runtime_cfg["attention_mode"]
        use_torch_compile = runtime_cfg["use_torch_compile"]
        device = runtime_cfg["device"]
        keep_model_loaded = runtime_cfg["keep_model_loaded"]
        keep_last_prompt = runtime_cfg["keep_last_prompt"]
        dry_run = runtime_cfg["dry_run"]

        keywords = remap_visual_keywords(collect_h3_keywords(options), target)
        overrides = remap_visual_overrides(collect_h3_overrides(options), target)
        placement = pick_option(options, H3_KEYWORD_PLACEMENT_KEY, PLACEMENT_BOTH)
        extra_tokens = coerce_token_dict(extra_tokens)

        system_prompt, user_prompt = resolve_finetune_messages(
            target=target,
            custom_prompt=custom_prompt,
            prepend_system_prompt=prepend_system_prompt,
            duration_s=duration_seconds,
            segment_s=segment_seconds,
            layout=prompt_layout,
            extra_tokens=extra_tokens,
        )
        system_prompt, user_prompt = attach_h3_keyword_instructions(
            system_prompt, user_prompt, keywords, placement, overrides
        )
        cache_preset = (
            f"{target}|dur={int(duration_seconds)}|"
            f"seg={int(segment_seconds)}|layout={prompt_layout}|"
            f"sys={prepend_system_prompt.strip()}|kw={keywords}|ovr={overrides}|place={placement}|tok={extra_tokens}|user={custom_prompt}"
        )
        kept_keywords = {key: value for key, value in keywords.items() if key not in overrides}
        if dry_run:
            generated = ""
            overridden = ""
            text = (
                "[dry run — model not loaded]\n\n"
                "SYSTEM\n------\n"
                f"{system_prompt}\n\n"
                "USER\n----\n"
                f"{user_prompt}"
            )
        else:
            result = self.run(
                model_name,
                quantization,
                target,
                custom_prompt,
                image,
                image2,
                frame_count,
                max_tokens,
                temperature,
                top_p,
                num_beams,
                repetition_penalty,
                seed,
                keep_model_loaded,
                attention_mode,
                use_torch_compile,
                device,
                keep_last_prompt,
                system_override=system_prompt,
                user_override=user_prompt,
                cache_preset=cache_preset,
            )
            generated = result[0] if result else ""
            overridden = apply_h3_section_overrides(generated, overrides)
            text = ensure_h3_keywords(overridden, kept_keywords, placement)

        notes = _finetune_debug_notes(
            options=options,
            cfg=cfg,
            model_name=model_name,
            quantization=quantization,
            attention_mode=attention_mode,
            device=device,
            keep_model_loaded=keep_model_loaded,
            keep_last_prompt=keep_last_prompt,
            image=image,
            image2=image2,
            keywords=keywords,
            kept_keywords=kept_keywords,
            overrides=overrides,
            placement=placement,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            generated=generated,
            overridden=overridden,
            text=text,
            custom_prompt=custom_prompt,
            extra_tokens=extra_tokens,
            dry_run=dry_run,
        )
        out_options = record_options_step(options, {}, "QwenVL-God Finetune", notes)
        return (text, out_options, format_debug_log(out_options, custom_prompt=custom_prompt, extra_tokens=extra_tokens))


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_Finetune": AILab_QwenVL_Finetune,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_Finetune": "QwenVL-God Finetune",
}
