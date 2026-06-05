# CLAUDE.md

**Granite Pipeline** — a single-shot Streamlit playground for analyzing text with IBM's [granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) (instruct) on Apple Silicon via `mlx-lm`. You provide text (paste, file upload, or a built-in sample), toggle which analyses to run, and get **summarization, topic detection, intent recognition, and sentiment** back. All four features are powered by prompting a single Granite model; the three classification features request JSON and are parsed defensively.

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

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `mlx-lm` — model loading and generation on Apple Silicon (mlx and transformers are transitive deps)
- `streamlit` — web UI
- `python-dotenv` — loads `HF_TOKEN` (and other env vars) from `.env` for local development

## Configuration

`pyproject.toml` — ruff lint isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`)

### Environment

`load_dotenv()` runs at the top of `streamlit_app.py`, before `load_model()` contacts the Hugging Face Hub.

- `HF_TOKEN` — optional Hugging Face access token (read scope). The Granite model is public, so it is not required; without it the HF Hub logs an "unauthenticated requests" warning and applies lower rate limits / slower Xet downloads.
- **Local**: set `HF_TOKEN` in `.env` (gitignored; copy from the committed `.env.example`). Loaded automatically by `python-dotenv`.
- **Deploy**: set `HF_TOKEN` as a real environment variable / platform secret. `load_dotenv()` does not override existing env vars and no-ops when no `.env` is present, so the same code works in both environments without shipping `.env`.

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
- `MAX_INPUT_TOKENS = 8192` — inputs longer than this are truncated (with a warning) before analysis.
- `TEMP = 0.0`, `TOP_P = 1.0`, `REPETITION_PENALTY = 1.2` — fixed decoding params. `temp=0.0` is greedy/deterministic, which keeps the JSON-emitting features reliably parseable. The repetition penalty is applied to **prose only** (it would fight the repeated structural tokens JSON requires).
- `FEATURES` — `list[dict]` registry; each entry has `key`, `label` (toggle), `tab_label` (result tab), `help` (toggle tooltip), `output` (`"prose"` or `"json"`), `max_tokens`, `system`, and `user_template` (formatted with `{text}`).
- `LABELS` — `{key: label}` derived from `FEATURES`.
- `SAMPLE_TEXTS` — `{name: text}` built-in samples for the Sample tab.

The four features: Summarization (prose, 256 tokens), Topic Detection and Intent Recognition (JSON, 256 tokens), Sentiment (JSON, 128 tokens). The three JSON features use IBM's "answer in JSON … `<schema>`" system-prompt pattern (reproduced verbatim, including the trailing newline); output is not guaranteed JSON, so it is parsed defensively.

### Session State

`st.session_state.results` — `dict | None`. Set on Run to `{"order": list[str], "data": {key: {"raw": str, "parsed": dict | None}}, "truncated": bool, "signature": (input_text, toggles)}`; `None` before the first run. Persists across reruns; the `signature` lets the results panel flag when the live input/toggles differ from the run.

### Layout

A full-width input section sits on top; below it the page splits into two columns (`st.columns(2)`).

- **Input** — `st.tabs(["Text", "Upload", "Sample"])`; the active input is resolved by precedence **Text > Upload > Sample** (first non-empty).
- **Left column** — a "Features" subheader over the four `st.toggle` widgets (default on; each description in its `help=` tooltip), with the **Run** button beneath (disabled until there is input and at least one feature is on).
- **Right column** — fixed result tabs: `JSON` plus one tab per feature, derived from `FEATURES` (`tab_label`). JSON shows the combined output; each feature tab renders its result (guarded by try/except), a "not enabled for this run" note if it was off, or a run prompt before the first run. An "inputs changed — click Run to refresh" note appears when the live input/toggles differ from the run. Rendered from `st.session_state.results`.

### Functions

- `load_model() -> tuple[nn.Module, PreTrainedTokenizerBase]` — loads model and tokenizer via `mlx_lm.load`, cached with `@st.cache_resource`.
- `truncate_to_tokens(text, tokenizer, max_tokens=MAX_INPUT_TOKENS) -> tuple[str, bool]` — truncates to a token budget; returns `(text, was_truncated)`.
- `parse_json_output(raw) -> dict | None` — returns the first JSON **object** found (tolerates surrounding prose/code fences via `JSONDecoder.raw_decode`); non-object JSON (lists, scalars) and unparseable input return `None`.
- `resolve_input(pasted, uploaded, sample) -> str` — resolves the active input by precedence (pasted > uploaded > sample); each candidate is stripped first, so a whitespace-only entry falls through.
- `run_feature(feature, text, model, tokenizer) -> dict` — builds the feature's chat-template prompt, runs `mlx_lm.generate` with a greedy `make_sampler` and — **for prose only** — a `make_logits_processors` repetition penalty; returns `{"raw": str, "parsed": dict | None}` (`parsed` only for JSON features).
- `render_result(key, result) -> None` — renders one feature's result with native components: prose for summary; a dataframe for topics (confidence as a `ProgressColumn`); `st.metric` for intent/sentiment, each with an optional rationale and a percent-formatted confidence (`_render_confidence`). Untrusted model output is guarded — non-list `topics` falls back to a message, `st.metric` values are coerced to strings, and JSON parse failures show the raw response.

### Generation

Each feature builds `[{"role": "system", ...}, {"role": "user", ...}]`, applies the chat template with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`, and passes the resulting string to `generate(...)` with the feature's `max_tokens`, `make_sampler(temp=TEMP, top_p=TOP_P)`, and a repetition-penalty `make_logits_processors` for prose features only (`logits_processors=None` for JSON). No "thinking" mode.

### Performance

- `@st.cache_resource` caches the model; it is loaded lazily on the first Run.
- MLX handles Apple Silicon (M-series) acceleration natively.
- Inputs over `MAX_INPUT_TOKENS` (8192) are truncated before analysis.
- Fixed greedy decoding (`temp=0.0`) keeps classification output deterministic and parseable.

### Error Handling

Unexpected exceptions during a run — and during per-feature rendering — are shown with `st.exception()`. JSON parse failures degrade gracefully to the raw response, and `render_result` guards/coerces untrusted model output before passing it to `st.metric` / `st.dataframe`.

## Tests

`tests/test_streamlit_app.py` — unit tests (mocked, no model download):

- `TestFeatures` — `FEATURES` order, required fields, prose-vs-JSON outputs, `LABELS` mapping, valid embedded JSON schemas, and the IBM-documented JSON system-prompt pattern (incl. the trailing newline)
- `TestParseJsonOutput` — plain / embedded / code-fenced JSON, first-of-multiple objects, recovery after stray braces, non-object JSON (arrays, scalars) → `None`, unparseable → `None`
- `TestResolveInput` — input precedence, whitespace stripping (incl. whitespace-only falling through to the next source), all-empty
- `TestTruncateToTokens` — short / long / boundary cases (uses `MAX_INPUT_TOKENS`), and the `add_special_tokens=False` encode flag
- `TestRunFeature` — prose vs JSON parsing, chat-template + `max_tokens` wiring, and decoding-param wiring (greedy sampler always; repetition penalty prose-only)
- `TestRenderResult` — `st.metric` string coercion for intent/sentiment, the non-list topics guard, and percent-formatted confidence (mocks `streamlit_app.st`)
