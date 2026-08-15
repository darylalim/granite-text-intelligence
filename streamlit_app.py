import json
import os
import re
from typing import Any, cast

import streamlit as st
from dotenv import load_dotenv
from mlx import nn
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

load_dotenv()  # populate HF_TOKEN from .env; deploy env vars take precedence

# 4-bit (affine, group_size 32) MLX conversion of ibm-granite/granite-4.1-8b:
# ~5.2 GB of weights vs ~16.8 GB for the bf16 build, same architecture and
# tokenizer. Quantization touches weights only, so the KV cache below is
# unaffected. Swap in "…-8b-bf16" (or "…-8b-8bit", ~9.4 GB) for more fidelity
# on a larger Mac.
MODEL_NAME = "mlx-community/granite-4.1-8b-4bit"

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
# `label` names the toggle; `tab_label` names the result tab. Every JSON feature
# also declares `localized_field`: the phrase naming its own free-text value
# ("every topic label"), which language_directive drops into the prompt. It
# lives here rather than as a branch inside language_directive so a newly added
# feature cannot silently inherit another feature's field name — omitting it
# raises a KeyError instead of telling the model to translate a field the
# schema does not have.
FEATURES: list[dict[str, Any]] = [
    {
        "key": "summary",
        "label": "Summarization",
        "tab_label": "Summary",
        "icon": ":material/summarize:",
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
        "label": "Topic detection",
        "tab_label": "Topics",
        "icon": ":material/label:",
        "help": "Identifies and ranks the main topics in your text.",
        "output": "json",
        "max_tokens": 256,
        "localized_field": "every topic label",
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
        "label": "Intent recognition",
        "tab_label": "Intents",
        "icon": ":material/flag:",
        "help": "Determines the primary intent expressed in your text.",
        "output": "json",
        "max_tokens": 256,
        "localized_field": "the rationale text",
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
        "icon": ":material/mood:",
        "help": "Classifies overall sentiment as positive, negative, neutral, or mixed.",
        "output": "json",
        "max_tokens": 128,
        "localized_field": "the rationale text",
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

# Localized output costs materially more tokens than the English-tuned
# `max_tokens` budgets, so structured output risks truncating mid-JSON. Every
# non-English target gets an enlarged budget — see `_effective_max_tokens`.
# max_tokens is a ceiling, so the extra headroom is free (generation stops at
# EOS). This deliberately covers Latin-script languages, not just CJK/Arabic: a
# German rationale overran the sentiment feature's 128-token budget in testing.
_LOCALIZED_TOKEN_MULTIPLIER = 2


st.set_page_config(
    page_title="Granite Text Intelligence",
    page_icon=":material/psychology:",
    # Wide, because the results panel is only half the page and carries five
    # icon-prefixed tabs; at the centered width they overflow into a scrolling
    # strip that hides whichever label is furthest from the active one.
    layout="wide",
)

st.session_state.setdefault("results", None)


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

    Three wording choices are deliberate, each a fix for observed misbehavior
    and each settled by an A/B against the live model (see CLAUDE.md):

    1. **Order.** The keep-English exception comes first and the localize
       requirement last, so the instruction nearest the generation point is the
       one we most need honored. The old order ("localize …, but keep …
       English") let the trailing English clause dominate and left rationale
       text in English.
    2. **Own field.** The JSON branch names the feature's own localizable field
       (rationale / topic label) rather than an abstract "all free-text field
       values", which the model applied to topic labels but not to rationale.
    3. **Unquoted.** That field name carries no quotes. Quoting it (`each topic
       "label"`) reliably collapsed Japanese labels into meaningless katakana —
       apparently read as a literal string to emit rather than a field to
       translate. The schema in the system prompt is what pins the actual key
       spelling, so the directive does not need to quote it.

    A fourth constraint is not a wording choice but a correctness one: the
    requirement is never phrased as a negative ("not in English"), because
    under LANGUAGE_AUTO the target may itself be English and contradict it.

    The "such as the sentiment label" aside is deliberately sent to *all three*
    JSON features even though only Sentiment has an enum. It reads as dead
    weight in the Topics and Intents prompts, but a second A/B replacing it
    with a feature-aware clause (keys-only where no enum exists) collapsed
    Japanese topic labels into meaningless katakana in both variants tried.
    The clause is load-bearing for reasons that are not obvious; leave it.
    """
    if language == LANGUAGE_ENGLISH:
        return ""
    target = (
        "the same language as the text above" if language == LANGUAGE_AUTO else language
    )
    if feature["output"] == "prose":
        # The trailing clause suppresses notes like "(Note: written in Japanese
        # as requested)", which the system prompt's "no preamble" already bars.
        return (
            f"\n\nWrite your entire response in {target}. Output only the "
            f"response itself, with no note or comment about the language used."
        )
    return (
        f"\n\nKeep every JSON key exactly as given in the schema, and keep "
        f"enumerated values (such as the sentiment label) in English. "
        f"Write {feature['localized_field']} in {target}."
    )


def _effective_max_tokens(feature: dict[str, Any], language: str) -> int:
    """Output-token budget for a feature, enlarged for any non-English output.

    The `max_tokens` values are tuned for English. Every other target costs more
    tokens for the same content: most acutely CJK/Arabic, but Latin-script
    languages too, since their accents and compounds fragment badly in a
    largely-English BPE vocabulary. A localized rationale or label set that runs
    past the budget truncates mid-object and fails to parse, so every
    non-English target — including "Match input", which can resolve to any of
    them — gets the larger ceiling. Since generation stops at EOS, the headroom
    is free: short output simply finishes early.
    """
    base = feature["max_tokens"]
    if language == LANGUAGE_ENGLISH:
        return base
    return base * _LOCALIZED_TOKEN_MULTIPLIER


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
    sampler = make_sampler(temp=TEMP)
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


# Color the sentiment metric value by its enum, via Streamlit's `:color[…]`
# markdown. Only Streamlit's built-in color names resolve here — an unrecognized
# one renders as literal text rather than raising, so TestThemeConfig pins these
# four against Streamlit's own set. The sentiment enum is fixed by the JSON
# schema and stays English, so this mapping is stable; orange (not the
# low-contrast yellow) is used for "mixed". Any out-of-enum label falls through
# to an uncolored value.
#
# "negative" stays red even though Streamlit's built-in themes make red the
# primary too (#FF4B4B — the Run button, the active tab underline, the selected
# sample chip), so the two share a hue family. Considered and kept: red for
# negative is near-universal, the alternatives are worse (orange is taken by
# "mixed", yellow is low-contrast on light), and separating them would mean
# reintroducing a custom theme the app deliberately does not ship.
_SENTIMENT_COLOR = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray",
    "mixed": "orange",
}


def _topic_rows(topics: Any) -> list[dict[str, Any]]:
    """Coerce the model's `topics` value into rows `st.dataframe` can render.

    A non-empty-list check is not enough, because three plausible malformations
    survive it and break the table in different ways:

    * a list of bare label strings (`["politics", "economy"]`) converts to a
      single column literally headed `value`; `column_config` names nothing in
      it, so the labels lose their header and the confidence bar disappears;
    * a list mixing objects with scalars raises `StreamlitAPIException`, which
      the caller's blind except degrades to "Could not render this result.";
    * a list of objects carrying no usable `label` (`[{}]`, or
      `[{"confidence": 0.9}]` from a truncated object) renders a table of blank
      rows instead of falling through to "No topics found."

    So bare strings are promoted to the documented `{"label": …}` shape, and an
    entry survives only if it has something to show in the Topic column. Both
    shapes are held to the *same* test, which is what lets a truthy return mean
    "there is a topic to render" — the caller's `if rows:` is the fallback's
    only guard. Dropping is not data loss for the user: the JSON tab always
    shows the raw response. A row missing `confidence` renders as an empty bar
    rather than raising.
    """
    if not isinstance(topics, list):
        return []
    rows: list[dict[str, Any]] = []
    for topic in topics:
        if isinstance(topic, dict) and str(topic.get("label", "")).strip():
            rows.append(topic)
        elif isinstance(topic, str) and topic.strip():
            rows.append({"label": topic})
    return rows


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
        # language=None is plain monospace. st.code defaults to language="python",
        # which syntax-highlights model prose as code on the one branch whose
        # whole purpose is showing what the model actually said — and claiming
        # "json" would be a guess, since this branch runs precisely because the
        # output did not parse. wrap_lines keeps a long single-line response
        # inside the results column instead of scrolling it sideways.
        st.code(raw, language=None, wrap_lines=True)
        return
    if key == "topics":
        rows = _topic_rows(parsed.get("topics"))
        if rows:
            st.dataframe(
                rows,
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
        sentiment = str(parsed.get("sentiment", "—"))
        color = _SENTIMENT_COLOR.get(sentiment.lower())
        st.metric("Sentiment", f":{color}[{sentiment}]" if color else sentiment)
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


st.title("Granite Text Intelligence")

# ---- Input: Text > Upload > Sample (first non-empty wins) ----
text_tab, upload_tab, sample_tab = st.tabs(
    [
        ":material/edit: Text",
        ":material/upload_file: Upload",
        ":material/dataset: Sample",
    ]
)
with text_tab:
    pasted = st.text_area(
        "Text",
        placeholder="Your text here...",
        height=200,
        label_visibility="collapsed",
        key="paste",
    )
with upload_tab:
    uploaded = st.file_uploader(
        "Upload a .txt or .md file", type=["txt", "md"], key="upload"
    )
    uploaded_text = (
        uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else ""
    )
    if uploaded_text:
        # A read-only preview, not an input. A disabled st.text_area is styled
        # at fadedText40 (Streamlit forces it through -webkit-text-fill-color)
        # and still occupies the tab order, so it reads as a broken field; a
        # fixed-height container scrolls the same way at full contrast and
        # leaves one fewer widget in the session.
        with st.container(height=150):
            st.text(uploaded_text, width="stretch")
with sample_tab:
    choice = st.segmented_control(
        "Pick a sample", list(SAMPLE_TEXTS), key="sample_select"
    )
    sample_text = SAMPLE_TEXTS.get(choice, "")
    if sample_text:
        # Read-only preview — see the Upload tab above for why this is not a
        # disabled st.text_area.
        with st.container(height=150):
            st.text(sample_text, width="stretch")

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
        icon=":material/play_arrow:",
        width="stretch",
        disabled=not (input_text and any(enabled.values())),
        key="run",
    )

with results_column:
    # Claim this panel's slots *before* the run, and fill them after. Streamlit
    # emits a delta as each st.* call executes and leaves the previous run's
    # elements on screen — faded to ~33% once the run outruns a short delay —
    # until the new run re-emits them. Creating the tabs after inference would
    # therefore leave the whole panel stale for the length of a 5.2 GB model
    # load plus up to four generations, including the now-actively-misleading
    # "Inputs changed since this run" note. The spinners below do not repaint
    # it: st.spinner enqueues on a *transient* cursor that never advances the
    # real delta path. Containers accept content out of order, so the bodies
    # below are unchanged — only the emission order is.
    status_slot = st.container()
    notice_slot = st.container()
    tabs = st.tabs(
        [
            ":material/data_object: JSON",
            *[f"{feature['icon']} {feature['tab_label']}" for feature in FEATURES],
        ]
    )
    json_tab = tabs[0]
    feature_tabs = {feature["key"]: tab for feature, tab in zip(FEATURES, tabs[1:])}

    if run:
        with status_slot:
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
            except Exception as exc:  # noqa: BLE001 (top-level run guard)
                st.exception(exc)

    results = cast("dict[str, Any] | None", st.session_state.results)
    if results is not None:
        with notice_slot:
            if results["truncated"]:
                st.warning(
                    f"Input was truncated to the first {MAX_INPUT_TOKENS} tokens.",
                    icon=":material/content_cut:",
                )
            current_signature = _run_signature(input_text, enabled, language)
            if results["signature"] != current_signature:
                st.info(
                    "Inputs changed since this run — click Run to refresh.",
                    icon=":material/sync:",
                )

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
            st.info(
                "Choose features and click Run to see results here.",
                icon=":material/play_circle:",
            )

    for key, tab in feature_tabs.items():
        with tab:
            if results is not None and key in results["data"]:
                try:
                    render_result(key, results["data"][key])
                except Exception as exc:  # noqa: BLE001 (untrusted model output shape)
                    st.warning("Could not render this result.")
                    st.exception(exc)
            elif results is None:
                st.info("Run to see results here.", icon=":material/play_circle:")
            else:
                st.caption(f"{LABELS[key]} was not enabled for this run.")
