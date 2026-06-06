import json
import os
import re
from typing import Any, cast

import mlx.nn as nn
import streamlit as st
from dotenv import load_dotenv
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

load_dotenv()  # populate HF_TOKEN from .env; deploy env vars take precedence

MODEL_NAME = "mlx-community/granite-4.1-8b-bf16"

# Granite 4.1's 128K context ceiling; configured caps are clamped to it.
MODEL_MAX_TOKENS = 131072

# Default input-token budget. Inputs longer than MAX_INPUT_TOKENS are truncated
# (with a warning) before analysis. The KV cache costs ~160 KB/token, so raising
# the cap raises memory and prefill latency — see CLAUDE.md. Larger Macs can opt
# into more via the MAX_INPUT_TOKENS env var.
_DEFAULT_MAX_INPUT_TOKENS = 16384


def _resolve_max_input_tokens() -> int:
    """Read MAX_INPUT_TOKENS from the environment, clamped to a safe range.

    Defaults to `_DEFAULT_MAX_INPUT_TOKENS` when unset; a non-integer *or
    non-positive* value falls back to that default (so a sign typo or `0` can't
    silently cap input to a single token). A valid value is clamped to
    `MODEL_MAX_TOKENS`.
    """
    raw = os.environ.get("MAX_INPUT_TOKENS")
    if raw is None:
        return _DEFAULT_MAX_INPUT_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_INPUT_TOKENS
    if value < 1:
        return _DEFAULT_MAX_INPUT_TOKENS
    return min(value, MODEL_MAX_TOKENS)


MAX_INPUT_TOKENS = _resolve_max_input_tokens()

# Fixed decoding params. temp=0.0 is greedy/deterministic, which keeps the
# JSON-emitting features reliably parseable. The repetition penalty is applied
# to prose only (see run_feature) — it would fight the repeated structural
# tokens that JSON requires.
TEMP = 0.0
TOP_P = 1.0
REPETITION_PENALTY = 1.2

# IBM Granite's documented JSON system-prompt pattern, reproduced verbatim
# (including the trailing newline) from the official granite-4.1 README/docs.
# Output is still not guaranteed JSON, so it is parsed defensively — see
# parse_json_output.
_JSON_SYSTEM = (
    "You are a helpful assistant that answers in JSON. Here's the json schema "
    "you must adhere to:\n<schema>\n{schema}\n</schema>\n"
)

# Each feature is fully described by its prompt, output kind, and token budget.
# `label` names the toggle; `tab_label` names the result tab.
FEATURES: list[dict[str, Any]] = [
    {
        "key": "summary",
        "label": "Summarization",
        "tab_label": "Summary",
        "help": "Generates a faithful, self-contained summary of your text.",
        "output": "prose",
        "max_tokens": 256,
        "system": (
            "You are a precise summarization assistant. Write a faithful, "
            "self-contained summary of the user's text. Do not add information that "
            "is not present. Output only the summary as plain prose, with no "
            "preamble, headings, or bullet labels."
        ),
        "user_template": "Summarize the following text in 3-5 sentences:\n\n<<<\n{text}\n>>>",
    },
    {
        "key": "topics",
        "label": "Topic Detection",
        "tab_label": "Topics",
        "help": "Identifies and ranks the main topics in your text.",
        "output": "json",
        "max_tokens": 256,
        "system": _JSON_SYSTEM.format(
            schema=(
                '{"type":"object","properties":{"topics":{"type":"array","items":'
                '{"type":"object","properties":{"label":{"type":"string"},'
                '"confidence":{"type":"number","minimum":0,"maximum":1}},'
                '"required":["label","confidence"]}}},"required":["topics"]}'
            )
        ),
        "user_template": (
            "Identify the main topics of the following text. Return 1 to 5 topics, "
            "most salient first. Output only JSON.\n\n<<<\n{text}\n>>>"
        ),
    },
    {
        "key": "intents",
        "label": "Intent Recognition",
        "tab_label": "Intents",
        "help": "Determines the primary intent expressed in your text.",
        "output": "json",
        "max_tokens": 256,
        "system": _JSON_SYSTEM.format(
            schema=(
                '{"type":"object","properties":{"intent":{"type":"string"},'
                '"confidence":{"type":"number","minimum":0,"maximum":1},'
                '"rationale":{"type":"string"}},"required":["intent","confidence"]}'
            )
        ),
        "user_template": (
            "Determine the primary intent expressed in the following text (what the "
            "author wants to happen or achieve). Output only JSON.\n\n<<<\n{text}\n>>>"
        ),
    },
    {
        "key": "sentiment",
        "label": "Sentiment",
        "tab_label": "Sentiment",
        "help": "Classifies overall sentiment as positive, negative, neutral, or mixed.",
        "output": "json",
        "max_tokens": 128,
        "system": _JSON_SYSTEM.format(
            schema=(
                '{"type":"object","properties":{"sentiment":{"type":"string",'
                '"enum":["positive","negative","neutral","mixed"]},'
                '"confidence":{"type":"number","minimum":0,"maximum":1},'
                '"rationale":{"type":"string"}},"required":["sentiment","confidence"]}'
            )
        ),
        "user_template": "Classify the overall sentiment of the following text. Output only JSON.\n\n<<<\n{text}\n>>>",
    },
]

LABELS: dict[str, str] = {feature["key"]: feature["label"] for feature in FEATURES}

SAMPLE_TEXTS: dict[str, str] = {
    "Product review": (
        "I bought these wireless earbuds last month and I'm honestly impressed. The "
        "battery easily lasts a full workday, pairing was instant, and the noise "
        "cancellation is better than headphones twice the price. My only gripe is that "
        "the touch controls are a little too sensitive. Overall, a great buy."
    ),
    "Support message": (
        "Hi, I was charged twice for my subscription this month and the second charge "
        "still hasn't been refunded after a week. I've already emailed support once "
        "with no reply. Can someone please look into this and refund the duplicate "
        "charge as soon as possible?"
    ),
    "News excerpt": (
        "The city council approved a plan on Tuesday to expand the downtown bike-lane "
        "network by 40 miles over the next three years. Supporters say the project will "
        "ease traffic congestion and cut emissions, while some local business owners "
        "worry about the temporary loss of street parking during construction."
    ),
}

# Granite 4.1's officially supported languages. Output can be localized to any of
# them, or "Match input" to mirror the analyzed text's language. Only free-text
# *values* are localized — JSON keys and enums stay English so parse_json_output
# and render_result keep working (see language_directive).
LANGUAGE_AUTO = "Match input"
LANGUAGE_ENGLISH = "English"
LANGUAGES: list[str] = [
    LANGUAGE_AUTO,
    LANGUAGE_ENGLISH,
    "German",
    "Spanish",
    "French",
    "Japanese",
    "Portuguese",
    "Arabic",
    "Czech",
    "Italian",
    "Korean",
    "Dutch",
    "Chinese",
]

# Languages whose output costs materially more tokens than the English-tuned
# `max_tokens` budgets, so structured output risks truncating mid-JSON. These
# (and "Match input", which can target any of them) get an enlarged budget — see
# `_effective_max_tokens`. max_tokens is a ceiling, so the extra headroom is free
# for short outputs (generation still stops at EOS).
_TOKEN_HEAVY_LANGUAGES = {"Japanese", "Chinese", "Korean", "Arabic"}
_LOCALIZED_TOKEN_MULTIPLIER = 2


st.set_page_config(page_title="Granite Pipeline")

if "results" not in st.session_state:
    st.session_state.results = None


@st.cache_resource
def load_model() -> tuple[nn.Module, TokenizerWrapper]:
    """Load model and tokenizer, cached for the session."""
    # load() returns a 2- or 3-tuple (the 3-tuple only when return_config=True,
    # which we don't pass), so its declared type is a union; narrow to the
    # 2-tuple we actually get.
    return cast("tuple[nn.Module, TokenizerWrapper]", load(MODEL_NAME))


def truncate_to_tokens(
    text: str, tokenizer: TokenizerWrapper, max_tokens: int = MAX_INPUT_TOKENS
) -> tuple[str, bool]:
    """Truncate text to at most max_tokens tokens. Returns (text, was_truncated)."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, False
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True), True


def parse_json_output(raw: str) -> dict[str, Any] | None:
    """Parse model output into the first JSON object found, or None.

    Tolerates surrounding prose/code fences, and ignores non-object JSON (lists,
    scalars) so callers can rely on the documented dict | None contract.
    """
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def resolve_input(pasted: str, uploaded: str, sample: str) -> str:
    """Resolve the active input by precedence: pasted > uploaded > sample.

    Each candidate is stripped first, so a whitespace-only entry falls through
    to the next source instead of suppressing it.
    """
    return pasted.strip() or uploaded.strip() or sample.strip()


def language_directive(feature: dict[str, Any], language: str) -> str:
    """Return a clause instructing the model which language to answer in.

    Localizes only free-text *values* (summary prose, rationale, topic labels).
    JSON keys and enumerated values (e.g. the sentiment label) must stay English
    so parse_json_output and render_result — which read results by English key —
    keep working. Returns "" for English (the prompts are already English).
    """
    if language == LANGUAGE_ENGLISH:
        return ""
    target = (
        "the same language as the text above" if language == LANGUAGE_AUTO else language
    )
    if feature["output"] == "prose":
        return f"\n\nWrite your entire response in {target}."
    return (
        f"\n\nWrite all free-text field values (such as rationale and topic "
        f"labels) in {target}, but keep every JSON key and any enumerated value "
        f"(such as the sentiment label) in English."
    )


def _effective_max_tokens(feature: dict[str, Any], language: str) -> int:
    """Output-token budget for a feature, enlarged for token-heavy languages.

    CJK/Arabic output costs more tokens than the English-tuned `max_tokens`, so a
    localized rationale/labels can truncate mid-JSON. Those languages — and
    "Match input", which can target any of them — get a larger ceiling; since
    generation stops at EOS, the headroom is free for short (e.g. English) output.
    """
    base = feature["max_tokens"]
    if language in _TOKEN_HEAVY_LANGUAGES or language == LANGUAGE_AUTO:
        return base * _LOCALIZED_TOKEN_MULTIPLIER
    return base


def run_feature(
    feature: dict[str, Any],
    text: str,
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    language: str = LANGUAGE_AUTO,
) -> dict[str, Any]:
    """Run one feature's prompt and return {"raw": str, "parsed": dict | None}.

    The language_directive is appended to the user turn (the system prompt stays
    verbatim, preserving IBM's documented JSON pattern).
    """
    user = feature["user_template"].format(text=text) + language_directive(
        feature, language
    )
    messages = [
        {"role": "system", "content": feature["system"]},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=TEMP, top_p=TOP_P)
    # Repetition penalty helps prose but harms JSON (it down-weights the repeated
    # braces, quotes, and keys the structured features rely on), so it is prose-only.
    logits_processors = (
        make_logits_processors(repetition_penalty=REPETITION_PENALTY)
        if feature["output"] == "prose"
        else None
    )
    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=_effective_max_tokens(feature, language),
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )
    parsed = parse_json_output(raw) if feature["output"] == "json" else None
    return {"raw": raw.strip(), "parsed": parsed}


def _render_confidence(parsed: dict[str, Any]) -> None:
    """Show the confidence value as a percentage when numeric, else verbatim."""
    confidence = parsed.get("confidence")
    if confidence is None:
        return
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        st.caption(f"Confidence: {confidence:.0%}")
    else:
        st.caption(f"Confidence: {confidence}")


def render_result(key: str, result: dict[str, Any]) -> None:
    """Render one feature's result using native components.

    Model output shape is untrusted, so values passed to widgets that reject
    odd types (st.metric, st.dataframe) are guarded/coerced.
    """
    raw, parsed = result["raw"], result["parsed"]
    if key == "summary":
        st.write(raw)
        return
    if parsed is None:
        st.warning("Could not parse JSON output; showing the raw response.")
        st.code(raw)
        return
    if key == "topics":
        topics = parsed.get("topics", [])
        if isinstance(topics, list) and topics:
            st.dataframe(
                topics,
                hide_index=True,
                column_config={
                    "label": st.column_config.TextColumn("Topic"),
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=1, format="percent"
                    ),
                },
            )
        else:
            st.write("No topics found.")
    elif key == "intents":
        st.metric("Intent", str(parsed.get("intent", "—")))
        _render_confidence(parsed)
        if parsed.get("rationale"):
            st.write(str(parsed["rationale"]))
    elif key == "sentiment":
        st.metric("Sentiment", str(parsed.get("sentiment", "—")))
        _render_confidence(parsed)
        if parsed.get("rationale"):
            st.write(str(parsed["rationale"]))


def _run_signature(
    input_text: str, enabled: dict[str, bool], language: str
) -> tuple[str, tuple[bool, ...], str]:
    """Identity of a run: the results panel flags them stale when this changes.

    Built once on Run and recomputed live each rerun; comparing the two is how
    the "inputs changed" note is driven. Kept in one place so the build- and
    compare-side never drift.
    """
    return (input_text, tuple(enabled[f["key"]] for f in FEATURES), language)


st.title("Granite Pipeline")
st.caption(
    "Get summarization, topics, intents, and sentiment based on your text input."
)

# ---- Input: Text > Upload > Sample (first non-empty wins) ----
text_tab, upload_tab, sample_tab = st.tabs(["Text", "Upload", "Sample"])
with text_tab:
    pasted = st.text_area(
        "Text",
        placeholder="Your text here...",
        height=200,
        label_visibility="collapsed",
    )
with upload_tab:
    uploaded = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
    uploaded_text = (
        uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else ""
    )
    if uploaded_text:
        st.text_area(
            "Uploaded",
            value=uploaded_text,
            height=150,
            disabled=True,
            label_visibility="collapsed",
        )
with sample_tab:
    choice = st.selectbox("Pick a sample", ["—", *SAMPLE_TEXTS])
    sample_text = SAMPLE_TEXTS.get(choice, "")
    if sample_text:
        st.text_area(
            "Sample",
            value=sample_text,
            height=150,
            disabled=True,
            label_visibility="collapsed",
        )

input_text = resolve_input(pasted, uploaded_text, sample_text)

# Output language is global (applies to every feature), so it sits with the input
# rather than in the per-feature column; constrained to a third of the width.
language_col, _ = st.columns([1, 2])
language = language_col.selectbox("Output language", LANGUAGES, key="language")

# ---- Features (left) and Results (right) ----
features_column, results_column = st.columns(2)

with features_column:
    st.subheader("Features")
    enabled: dict[str, bool] = {
        feature["key"]: st.toggle(
            feature["label"],
            value=True,
            help=feature["help"],
            key=f"feature_{feature['key']}",
        )
        for feature in FEATURES
    }
    run = st.button(
        "Run",
        type="primary",
        disabled=not (input_text and any(enabled.values())),
        key="run",
    )

with results_column:
    if run:
        try:
            with st.spinner("Loading model…"):
                model, tokenizer = load_model()
            text, was_truncated = truncate_to_tokens(input_text, tokenizer)
            data: dict[str, Any] = {}
            for feature in FEATURES:
                if not enabled[feature["key"]]:
                    continue
                with st.spinner(f"Running {feature['label']}…"):
                    data[feature["key"]] = run_feature(
                        feature, text, model, tokenizer, language
                    )
            st.session_state.results = {
                "order": [f["key"] for f in FEATURES if enabled[f["key"]]],
                "data": data,
                "truncated": was_truncated,
                "signature": _run_signature(input_text, enabled, language),
            }
        except Exception as exc:
            st.exception(exc)

    results = cast("dict[str, Any] | None", st.session_state.results)
    if results is not None:
        if results["truncated"]:
            st.warning(f"Input was truncated to the first {MAX_INPUT_TOKENS} tokens.")
        current_signature = _run_signature(input_text, enabled, language)
        if results["signature"] != current_signature:
            st.info("Inputs changed since this run — click Run to refresh.")

    tabs = st.tabs(["JSON", *[feature["tab_label"] for feature in FEATURES]])
    json_tab = tabs[0]
    feature_tabs = {feature["key"]: tab for feature, tab in zip(FEATURES, tabs[1:])}

    with json_tab:
        if results is not None and results["data"]:
            result_data = results["data"]
            st.json(
                {
                    key: result_data[key]["parsed"]
                    if result_data[key]["parsed"] is not None
                    else result_data[key]["raw"]
                    for key in results["order"]
                }
            )
        else:
            st.info("Choose features and click Run to see results here.")

    for key, tab in feature_tabs.items():
        with tab:
            if results is not None and key in results["data"]:
                try:
                    render_result(key, results["data"][key])
                except Exception as exc:  # untrusted model output shape
                    st.warning("Could not render this result.")
                    st.exception(exc)
            elif results is None:
                st.info("Run to see results here.")
            else:
                st.caption(f"{LABELS[key]} was not enabled for this run.")
