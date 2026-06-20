# CLAUDE.md

**Granite Text Intelligence** — a single-shot Streamlit playground for analyzing text with IBM's [granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) (instruct) on Apple Silicon via `mlx-lm`. You provide text (paste, file upload, or a built-in sample), toggle which analyses to run, and get **summarization, topic detection, intent recognition, and sentiment** back. All four features are powered by prompting a single Granite model; the three classification features request JSON and are parsed defensively. Output can be localized to any of Granite's 12 supported languages (default: match the input).

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `mlx-lm` (pinned `>=0.31,<0.32`) — model loading and generation on Apple Silicon; mlx and transformers are transitive deps. The `<0.32` cap guards against API drift, since decoding params pass through `generate(**kwargs)` rather than typed parameters; exact reproducibility is handled by `uv.lock`.
- `streamlit` — web UI
- `python-dotenv` — loads `HF_TOKEN` (and other env vars) from `.env` for local development

## Configuration

`pyproject.toml` — ruff lint isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`)

### Environment

`load_dotenv()` runs at the top of `streamlit_app.py`, before `load_model()` contacts the Hugging Face Hub.

- `HF_TOKEN` — optional Hugging Face access token (read scope). The Granite model is public, so it is not required; without it the HF Hub logs an "unauthenticated requests" warning and applies lower rate limits / slower Xet downloads.
- **Local**: set `HF_TOKEN` in `.env` (gitignored; copy from the committed `.env.example`). Loaded automatically by `python-dotenv`.
- **Deploy**: set `HF_TOKEN` as a real environment variable / platform secret. `load_dotenv()` does not override existing env vars and no-ops when no `.env` is present, so the same code works in both environments without shipping `.env`.

- `MAX_INPUT_TOKENS` — optional integer input-token budget (default `16384`, clamped to `MODEL_MAX_TOKENS = 131072`). Resolved once at import by `_resolve_max_input_tokens()` (non-integer, zero, or negative → default, so a sign typo can't silently cap input to one token). Raising it increases context but also memory (~160 KB KV cache/token) and prefill latency, so the default stays conservative and larger-RAM Macs opt into more. Set the same way as `HF_TOKEN` (`.env` locally / real env var on deploy).

## Architecture

`streamlit_app.py` — single-file app. Single-shot flow: one input → run the selected features → show results. Default Streamlit theme, native components, no sidebar.

### Model

```python
from mlx_lm import generate, load
model, tokenizer = load("mlx-community/granite-4.1-8b-bf16")
```

`load_model()` is called lazily inside the Run handler (not at module import), so importing the module — e.g. in tests — does not load the ~16.8 GB model.

### Constants

- `MODEL_NAME` — `mlx-community/granite-4.1-8b-bf16`.
- `MODEL_MAX_TOKENS = 131072` — Granite 4.1's 128K context ceiling; configured caps are clamped to it.
- `MAX_INPUT_TOKENS` — input-token budget; inputs longer than this are truncated (with a warning) before analysis. Defaults to `16384`, overridable via the `MAX_INPUT_TOKENS` env var (see Environment), resolved by `_resolve_max_input_tokens()`.
- `TEMP = 0.0`, `REPETITION_PENALTY = 1.2` — fixed decoding params. `temp=0.0` is greedy/deterministic (`make_sampler` returns argmax), which keeps the JSON-emitting features reliably parseable; `top_p` is left at its default since it has no effect under greedy decoding. The repetition penalty is applied to **prose only** (it would fight the repeated structural tokens JSON requires).
- `FEATURES` — `list[dict]` registry; each entry has `key`, `label` (toggle), `tab_label` (result tab), `help` (toggle tooltip), `output` (`"prose"` or `"json"`), `max_tokens`, `system`, and `user_template` (formatted with `{text}`).
- `LABELS` — `{key: label}` derived from `FEATURES`.
- `SAMPLE_TEXTS` — `{name: text}` built-in samples for the Sample tab.
- `LANGUAGES` — Granite 4.1's 12 officially supported output languages, led by the named constants `LANGUAGE_AUTO = "Match input"` (the default) and `LANGUAGE_ENGLISH = "English"`. Drives the **Output language** selectbox. Input is multilingual regardless; this controls output language.
- `_TOKEN_HEAVY_LANGUAGES` (Japanese/Chinese/Korean/Arabic) — output in these scripts costs more tokens, so `_effective_max_tokens` multiplies a feature's `max_tokens` (by `_LOCALIZED_TOKEN_MULTIPLIER = 2`) for them and for "Match input". `max_tokens` is a ceiling, so the headroom is free for short (e.g. English) output.

The four features: Summarization (prose, 256 tokens), Topic Detection and Intent Recognition (JSON, 256 tokens), Sentiment (JSON, 128 tokens). The three JSON features use IBM's "answer in JSON … `<schema>`" system-prompt pattern (reproduced verbatim, including the trailing newline); output is not guaranteed JSON, so it is parsed defensively.

> **Design decision (2026-06-05):** Granite's native tool-calling (`tools=` → `<tool_call>` blocks) was evaluated as an alternative structured-output channel for the classification features and **rejected**. An A/B over a 55-case adversarial corpus showed no benefit — schema-valid **53/53 tied**, accuracy **45 vs 44** (noise), diverging only on prompt-injection where tool-calling was marginally *worse*. The uniform JSON-prompt approach was kept; don't re-litigate without a material model/task change.

### Session State

`st.session_state.results` — `dict | None`. Set on Run to `{"order": list[str], "data": {key: {"raw": str, "parsed": dict | None}}, "truncated": bool, "signature": (input_text, toggles, language)}`; `None` before the first run. Persists across reruns; the `signature` lets the results panel flag when the live input/toggles/language differ from the run. Both the stored and live signatures are built by `_run_signature(input_text, enabled, language)` so the two sides can't drift.

### Layout

A full-width input section sits on top; below it the page splits into two columns (`st.columns(2)`).

- **Input** — `st.tabs(["Text", "Upload", "Sample"])`; the active input is resolved by precedence **Text > Upload > Sample** (first non-empty). Directly beneath the input (full-width, above the column split) is the **Output language** selectbox (`LANGUAGES`, default "Match input"), width-constrained to ~1/3 via `st.columns([1, 2])` — it's a global setting, so it sits with the input rather than in the per-feature column.
- **Left column** — a "Features" subheader over the four `st.toggle` widgets (default on; each description in its `help=` tooltip), with the **Run** button beneath (disabled until there is input and at least one feature is on).

Interactive widgets carry stable `key=`s so `AppTest` can address them by key rather than positional index: `paste` (text area), `upload` (file uploader), `sample_select` (sample selectbox), `feature_<key>` (per-feature toggles, e.g. `feature_summary`), `language` (output-language selectbox), and `run` (the Run button).
- **Right column** — fixed result tabs: `JSON` plus one tab per feature, derived from `FEATURES` (`tab_label`). JSON shows the combined output; each feature tab renders its result (guarded by try/except), a "not enabled for this run" note if it was off, or a run prompt before the first run. An "inputs changed — click Run to refresh" note appears when the live input/toggles/language differ from the run. Rendered from `st.session_state.results`.

### Functions

- `load_model() -> tuple[nn.Module, TokenizerWrapper]` — loads model and tokenizer via `mlx_lm.load`, cached with `@st.cache_resource`.
- `truncate_to_tokens(text, tokenizer, max_tokens=MAX_INPUT_TOKENS) -> tuple[str, bool]` — truncates to a token budget; returns `(text, was_truncated)`.
- `parse_json_output(raw) -> dict | None` — returns the first JSON **object** found (tolerates surrounding prose/code fences via `JSONDecoder.raw_decode`); non-object JSON (lists, scalars) and unparseable input return `None`.
- `resolve_input(pasted, uploaded, sample) -> str` — resolves the active input by precedence (pasted > uploaded > sample); each candidate is stripped first, so a whitespace-only entry falls through.
- `language_directive(feature, language) -> str` — clause appended to the **user** turn telling the model which language to answer in. Localizes only free-text *values* (summary prose, rationale, topic labels); JSON keys and enums (e.g. the sentiment label) stay English so `parse_json_output` / `render_result` keep working. Returns `""` for `"English"`; `"Match input"` mirrors the analyzed text's language.
- `_effective_max_tokens(feature, language) -> int` — the feature's output budget, doubled for `_TOKEN_HEAVY_LANGUAGES` / "Match input" so localized JSON doesn't truncate mid-object; the base `max_tokens` otherwise.
- `run_feature(feature, text, model, tokenizer, language=LANGUAGE_AUTO) -> dict` — builds the feature's chat-template prompt (user turn carries the `language_directive`; system prompt stays verbatim), runs `mlx_lm.generate` with `_effective_max_tokens(feature, language)`, a greedy `make_sampler`, and — **for prose only** — a `make_logits_processors` repetition penalty; returns `{"raw": str, "parsed": dict | None}` (`parsed` only for JSON features).
- `_run_signature(input_text, enabled, language) -> tuple` — the run's identity `(input_text, toggle states, language)`; built on Run and recomputed live so the results panel can flag stale results. Single source so build/compare sides can't diverge.
- `render_result(key, result) -> None` — renders one feature's result with native components: prose for summary; a dataframe for topics (confidence as a `ProgressColumn`); `st.metric` for intent/sentiment, each with an optional rationale and a percent-formatted confidence (`_render_confidence`). Untrusted model output is guarded — non-list `topics` falls back to a message, `st.metric` values are coerced to strings, and JSON parse failures show the raw response.

### Generation

Each feature builds `[{"role": "system", ...}, {"role": "user", ...}]`, applies the chat template with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`, and passes the resulting string to `generate(...)` with `_effective_max_tokens(feature, language)` (the base budget, enlarged for token-heavy output languages), `make_sampler(temp=TEMP)`, and a repetition-penalty `make_logits_processors` for prose features only (`logits_processors=None` for JSON). No "thinking" mode.

### Performance

- `@st.cache_resource` caches the model; it is loaded lazily on the first Run.
- MLX handles Apple Silicon (M-series) acceleration natively.
- Inputs over `MAX_INPUT_TOKENS` (default 16384, env-configurable) are truncated before analysis.
- Fixed greedy decoding (`temp=0.0`) keeps classification output deterministic and parseable.

### Error Handling

Unexpected exceptions during a run — and during per-feature rendering — are shown with `st.exception()`. JSON parse failures degrade gracefully to the raw response, and `render_result` guards/coerces untrusted model output before passing it to `st.metric` / `st.dataframe`.

## Tests

`tests/test_streamlit_app.py` — unit tests (mocked, no model download). Data-driven cases use `@pytest.mark.parametrize` (each `pytest.param` carries an `id=` so failures are self-labeling); `TestRunFeature` shares a `tokenizer` fixture:

- `TestFeatures` — `FEATURES` order, required fields, prose-vs-JSON outputs, `LABELS` mapping, valid embedded JSON schemas, and the IBM-documented JSON system-prompt pattern (incl. the trailing newline)
- `TestParseJsonOutput` — plain / embedded / code-fenced JSON, first-of-multiple objects, recovery after stray braces, non-object JSON (arrays, scalars) → `None`, unparseable → `None`
- `TestResolveInput` — input precedence, whitespace stripping (incl. whitespace-only falling through to the next source), all-empty
- `TestTruncateToTokens` — short / long / boundary cases (uses `MAX_INPUT_TOKENS`), and the `add_special_tokens=False` encode flag
- `TestRunFeature` — prose vs JSON parsing, chat-template + `max_tokens` wiring, decoding-param wiring (greedy sampler always; repetition penalty prose-only), and the `language` directive landing on the user turn (and absent for English)
- `TestLanguageDirective` — `English` → empty, prose targets the language, JSON localizes values but keeps keys/enums English, and `Match input` uses the relative phrase
- `TestResolveMaxInputTokens` — env-var override, default when unset, non-integer/non-positive fallback, clamp to `MODEL_MAX_TOKENS`, and the pinned default/ceiling values
- `TestEffectiveMaxTokens` — base budget for English/Latin languages; doubled for token-heavy scripts and "Match input"
- `TestRenderResult` — `st.metric` string coercion for intent/sentiment, the non-list topics guard, and percent-formatted confidence (mocks `streamlit_app.st`)

`tests/test_app_ui.py` — integration tests via Streamlit's `AppTest` (`streamlit.testing.v1`), driving the imperative UI block headlessly. The Run path is mocked at the **`mlx_lm` boundary** (`patch("mlx_lm.load" / "mlx_lm.generate")`) — `streamlit_app` re-execs `from mlx_lm import generate, load` on every run, so the imports bind to the mocks; an autouse fixture clears `st.cache_resource` between tests so each test's mock is used. Widgets are addressed by `key=`, not index.

- `TestInitialRender` — no model needed: Run disabled with no input, the four toggles default-on with correct labels, the Output language selectbox defaulting to "Match input", and the pre-run prompt / `results is None`
- `TestRunInteraction` — Run enables once text is entered; the mocked Run path populates `session_state.results`, renders the Sentiment `st.metric` and percent confidence, shows the "not enabled for this run" note for toggled-off features, flags "inputs changed" after a post-run edit (input *or* output-language change), and warns on truncated input
