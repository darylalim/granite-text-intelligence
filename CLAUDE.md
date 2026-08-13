# CLAUDE.md

**Granite Text Intelligence** — a Streamlit application for **summarization, topic, intent, and sentiment analysis** using IBM's [granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) (instruct) on Apple Silicon with MLX (`mlx-lm`). A single-shot playground: you provide text (paste, file upload, or a built-in sample), toggle which analyses to run, and get the results back. All four features are powered by prompting a single Granite model; the three classification features request JSON and are parsed defensively. Output can be localized to any of Granite's 12 supported languages (default: match the input).

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .` — reaches **Markdown as well as Python**: ruff ≥ 0.16 formats the code inside `python`-tagged fenced blocks, so a mis-spaced snippet in `CLAUDE.md` fails `--check` in CI exactly like a `.py` file would. **Formatting only** — `ruff check` skips Markdown outright (it reports "No Python files found" and exits 0), so an unused or unsorted import inside a fence is *not* linted. `README.md` is untouched today; its fences are all `bash`.
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

These same four checks run in CI (`.github/workflows/ci.yml`) on every push to `main` and pull request, and locally as Claude Code hooks (`.claude/settings.json`).

No dev tool carries a version constraint in `pyproject.toml`. `uv.lock` still pins exact versions and CI runs `uv sync --locked`, so any given build is reproducible — but nothing bounds what the *next* `uv lock --upgrade` pulls in, which means a **linter upgrade can redefine the gate without a line of our code changing**: ruff 0.16 expanded its default rule set (see Error Handling for the `BLE001` fallout) and widened the formatter to Markdown, both of which landed as gate failures on a routine `uv lock --upgrade`. Treat a ruff/ty major as its own reviewable change rather than incidental lockfile noise.

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `mlx-lm` (pinned `>=0.31,<0.32`) — model loading and generation on Apple Silicon; mlx and transformers are transitive deps. The `<0.32` cap guards against API drift, since decoding params pass through `generate(**kwargs)` rather than typed parameters; exact reproducibility is handled by `uv.lock`.
  - **The cap binds `mlx-lm`, not `mlx`.** mlx-lm declares `mlx` with *no* version bound, so the compute library floats free of the guard above — and since 0.31.3 is mlx-lm's latest release, the cap is not currently binding on anything. As of 2026-08-13 the lock pairs mlx-lm 0.31.3 with mlx 0.32.0, a combination newer than anything upstream shipped together (the mlx-lm sdist predates mlx 0.32.0). That pairing is **verified by real-inference smoke test**, not by CI: all four features returned sensible output, all three JSON features parsed, and a localized German run localized its rationale without overrunning Sentiment's doubled budget while keeping keys and the enum English.
  - The test suite structurally **cannot** cover this. Every test mocks at the `mlx_lm` boundary (see Tests), which is the right call — real inference in CI would mean a 5.2 GB download per run — but it means no test executes a line of `mlx`, and the only real contact in CI is the module-level `from mlx import nn`. So when a `uv lock --upgrade` moves `mlx`, `mlx-lm`, `transformers`, or `huggingface-hub`, run the model by hand against the local cache (`HF_HUB_OFFLINE=1`) before trusting four green gates. Cold download and `HF_TOKEN` auth stay unexercised either way — a warm-cache smoke test says nothing about them.
- `streamlit` (pinned `>=1.57`) — web UI. The floor is load-bearing: the IBM Carbon theme's per-mode `[theme.light]`/`[theme.dark]` blocks, `width="stretch"`, and Material Symbol icons need a recent Streamlit (older versions only *warn* on unrecognized theme keys, silently degrading the theme). Exact version pinned by `uv.lock`.
- `python-dotenv` — loads `HF_TOKEN` (and other env vars) from `.env` for local development

## Configuration

`pyproject.toml` — ruff lint (`extend-select = ["I"]` turns on isort import sorting atop ruff's defaults; `combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`)

`LICENSE` — Apache License 2.0 (full canonical text, copyright Daryl Lim); the project's **code** is released under it. Declared to tooling in `pyproject.toml` via PEP 639 SPDX fields (`license = "Apache-2.0"`, `license-files = ["LICENSE"]`) and summarized in the README's License section. The IBM Granite model is licensed separately (also Apache 2.0) and downloaded at runtime, not vendored here.

`.python-version` — pins the project interpreter to `3.12` (via `uv python pin`), which `uv sync` / `uv run` honor automatically. Keeps the version you run and test against aligned with the `requires-python = ">=3.12"` floor and the ty type-check target, instead of letting uv auto-select the newest installed Python (e.g. 3.13).

`.github/workflows/ci.yml` — GitHub Actions CI: runs the four Commands (lint, format `--check`, typecheck, test) under `uv sync --locked` on every push to `main` and pull request. Pinned to a `macos-14` (Apple Silicon) runner — required, since `mlx`/`mlx-metal` are `sys_platform == 'darwin'` in `uv.lock` and `streamlit_app.py` imports `mlx` at module top, so a Linux runner couldn't even collect the tests. `TestCIWorkflow` guards this config against drift.

`.claude/settings.json` — project-shared Claude Code hooks: `ruff` format + lint-fix on each edited `.py` file and `ruff format` on each edited `.md` file (format only — `ruff check` is a no-op on Markdown; the `.md` branch exists because ruff ≥ 0.16 made doc snippets a formatted artifact the Stop hook and CI enforce), a guard blocking edits to `.env` / `.env.*` (the committed `.env.example` is exempt so the template stays editable) / `secrets.toml` / `uv.lock`, and the full quality gate on Stop. Personal overrides go in the gitignored `.claude/settings.local.json`.

`.streamlit/config.toml` — an IBM Carbon-inspired theme: IBM Plex Sans/Mono (loaded from Google Fonts, so no local font files) over IBM's Blue 60 (`#0f62fe`) primary. Shared font/radius live in `[theme]`; per-mode colors in separate `[theme.light]` / `[theme.dark]` blocks — defining **both** is what surfaces the light/dark toggle in the app's settings menu (a lone `[theme]` locks one mode). Streamlit only *warns* on an unrecognized theme key (a casing typo silently disables that style), so `TestThemeConfig` cross-checks every key against `streamlit.config.get_config_options()`.

### Environment

`load_dotenv()` runs at the top of `streamlit_app.py`, before `load_model()` contacts the Hugging Face Hub.

- `HF_TOKEN` — optional Hugging Face access token (read scope). The Granite model is public, so it is not required; without it the HF Hub logs an "unauthenticated requests" warning and applies lower rate limits / slower Xet downloads.
- **Local**: set `HF_TOKEN` in `.env` (gitignored; copy from the committed `.env.example`). Loaded automatically by `python-dotenv`.
- **Deploy**: set `HF_TOKEN` as a real environment variable / platform secret. `load_dotenv()` does not override existing env vars and no-ops when no `.env` is present, so the same code works in both environments without shipping `.env`.

- `MAX_INPUT_TOKENS` — optional integer input-token budget (default `16384`, clamped to `MODEL_MAX_TOKENS = 131072`). Resolved once at import by `_resolve_max_input_tokens()` (non-integer, zero, or negative → default, so a sign typo can't silently cap input to one token). Raising it increases context but also memory (~160 KB KV cache/token) and prefill latency, so the default stays conservative and larger-RAM Macs opt into more. Set the same way as `HF_TOKEN` (`.env` locally / real env var on deploy).

## Architecture

`streamlit_app.py` — single-file app. Single-shot flow: one input → run the selected features → show results. An IBM Carbon-inspired theme (`.streamlit/config.toml`; see Configuration), native components, Material Symbol icons, no sidebar.

### Model

```python
from mlx_lm import generate, load

model, tokenizer = load("mlx-community/granite-4.1-8b-4bit")
```

`load_model()` is called lazily inside the Run handler (not at module import), so importing the module — e.g. in tests — does not load the ~5.2 GB model.

### Constants

- `MODEL_NAME` — `mlx-community/granite-4.1-8b-4bit`, a 4-bit (affine, `group_size` 32) MLX conversion of `ibm-granite/granite-4.1-8b`: ~5.2 GB of weights vs ~16.8 GB for the bf16 build, at identical architecture, context, and tokenizer. Quantization touches weights only, so the ~160 KB/token KV cache is unchanged. `…-8b-8bit` (~9.4 GB) and `…-8b-bf16` (~16.8 GB) are the higher-fidelity swaps for larger Macs.
- `MODEL_MAX_TOKENS = 131072` — Granite 4.1's 128K context ceiling; configured caps are clamped to it.
- `MAX_INPUT_TOKENS` — input-token budget; inputs longer than this are truncated (with a warning) before analysis. Defaults to `16384`, overridable via the `MAX_INPUT_TOKENS` env var (see Environment), resolved by `_resolve_max_input_tokens()`.
- `TEMP = 0.0`, `REPETITION_PENALTY = 1.2` — fixed decoding params. `temp=0.0` is greedy/deterministic (`make_sampler` returns argmax), which keeps the JSON-emitting features reliably parseable; `top_p` is left at its default since it has no effect under greedy decoding. The repetition penalty is applied to **prose only** (it would fight the repeated structural tokens JSON requires).
- `FEATURES` — `list[dict]` registry; each entry has `key`, `label` (toggle), `tab_label` (result tab), `icon` (the result tab's Material Symbol shortcode, e.g. `:material/mood:`), `help` (toggle tooltip), `output` (`"prose"` or `"json"`), `max_tokens`, `system`, and `user_template` (formatted with `{text}`). JSON entries additionally carry `localized_field` — the phrase naming that feature's own free-text value (`"every topic label"`, `"the rationale text"`) that `language_directive` drops into the prompt. It lives in the registry rather than as a branch inside `language_directive` so a new feature can't inherit another's field name; `language_directive` indexes it directly, so omitting it raises a `KeyError` instead of instructing the model to translate a field the schema lacks. `TestFeatures` also checks the phrase names a field the feature's own schema declares.
- `LABELS` — `{key: label}` derived from `FEATURES`.
- `SAMPLE_TEXTS` — `{name: text}` built-in samples for the Sample tab.
- `LANGUAGES` — the **Output language** selectbox list (13 entries): the `LANGUAGE_AUTO = "Match input"` sentinel (the default, not itself a language), followed by Granite 4.1's 12 officially supported output languages — the first of which is `LANGUAGE_ENGLISH = "English"`. Input is multilingual regardless; this controls output language.
- `_LOCALIZED_TOKEN_MULTIPLIER = 2` — every **non-English** output language (including "Match input") gets `max_tokens × 2` via `_effective_max_tokens`, since localized output costs more tokens than the English-tuned budgets and a JSON feature that overruns truncates mid-object and fails to parse. `max_tokens` is a ceiling, so the headroom is free for short output. This deliberately covers Latin-script languages, not just CJK/Arabic — a `_TOKEN_HEAVY_LANGUAGES` set previously narrowed it to Japanese/Chinese/Korean/Arabic, which held only while the rationale field was (incorrectly) staying English; once `language_directive` was fixed, a localized German rationale overran Sentiment's 128-token budget.

The four features: Summarization (prose, 256 tokens), Topic detection and Intent recognition (JSON, 256 tokens), Sentiment (JSON, 128 tokens). The three JSON features use IBM's "answer in JSON … `<schema>`" system-prompt pattern (reproduced verbatim, including the trailing newline); output is not guaranteed JSON, so it is parsed defensively.

> **Design decision (2026-08-12):** `language_directive`'s JSON wording was settled by an A/B of five phrasings against the live model (greedy decoding, so single runs are deterministic), scored on Japanese topic labels and rationale localization. Two independent variables emerged. **Clause order** decides whether `rationale` localizes at all: both candidates ending on the keep-English clause left it in English; all three ending on the localize clause localized it. **Quoting the field name** wrecks topic labels: `each topic "label"` collapsed them into meaningless katakana (`メーマイファーション`), while the otherwise-identical unquoted `every topic label` produced clean output — the quotes appear to read as a literal string to emit rather than a field to translate. Shipped: exception first, requirement last, field named unquoted. A knock-on: once the rationale actually localized, a German one overran Sentiment's 128-token budget and truncated mid-object, which is why `_effective_max_tokens` now widens for *every* non-English language rather than a CJK/Arabic subset. Don't "simplify" this wording without re-running the A/B.
>
> A follow-up A/B (same day) tried to remove what looks like dead weight: all three JSON features are told to "keep enumerated values (such as the sentiment label) in English", but only Sentiment has an enum, and in the Topics prompt the word *label* collides with the field being localized. **Both** feature-aware replacements — a keys-only clause where no enum exists, in two phrasings — collapsed Japanese topic labels into meaningless katakana (`モーロウングイス`, `ビルグイングイン`) while leaving German *better*. The clause is load-bearing for non-obvious reasons and was kept verbatim on all three features. Note the automated scorer passed all three candidates: its "is it localized?" check was a CJK codepoint test, and degenerate katakana is still CJK. Any future A/B here needs a meaningfulness check, not just a script check.

> **Design decision (2026-06-05):** Granite's native tool-calling (`tools=` → `<tool_call>` blocks) was evaluated as an alternative structured-output channel for the classification features and **rejected**. An A/B over a 55-case adversarial corpus showed no benefit — schema-valid **53/53 tied**, accuracy **45 vs 44** (noise), diverging only on prompt-injection where tool-calling was marginally *worse*. The uniform JSON-prompt approach was kept; don't re-litigate without a material model/task change. **Caveat (2026-08-12):** those numbers were measured while `MODEL_NAME` was the bf16 build and have **not** been re-run on the 4-bit default now shipped. Supporting but weaker evidence exists for 4-bit — 12/12 parses across the three built-in samples, plus 12 more across Japanese/German/"Match input" — but that is not the 55-case adversarial corpus. Treat "53/53" as a bf16-era result; re-run the corpus before citing it as evidence about the shipped model.

### Session State

`st.session_state.results` — `dict | None`. Set on Run to `{"order": list[str], "data": {key: {"raw": str, "parsed": dict | None}}, "truncated": bool, "signature": (input_text, toggles, language)}`; `None` before the first run. Persists across reruns; the `signature` lets the results panel flag when the live input/toggles/language differ from the run. Both the stored and live signatures are built by `_run_signature(input_text, enabled, language)` so the two sides can't drift.

### Layout

A full-width input section sits on top; below it the page splits into two columns (`st.columns(2)`).

- **Input** — `st.tabs(["Text", "Upload", "Sample"])` (each label prefixed with a Material Symbol icon); the active input is resolved by precedence **Text > Upload > Sample** (first non-empty). Directly beneath the input (full-width, above the column split) is the **Output language** selectbox (`LANGUAGES`, default "Match input"), width-constrained to ~1/3 via `st.columns([1, 2])` — it's a global setting, so it sits with the input rather than in the per-feature column.
- **Left column** — a "Features" subheader over the four `st.toggle` widgets (default on; each description in its `help=` tooltip), with the full-width **Run** button beneath (a `:material/play_arrow:` icon; `width="stretch"`; disabled until there is input and at least one feature is on).

Interactive widgets carry stable `key=`s so `AppTest` can address them by key rather than positional index: `paste` (text area), `upload` (file uploader), `sample_select` (sample segmented control), `feature_<key>` (per-feature toggles, e.g. `feature_summary`), `language` (output-language selectbox), and `run` (the Run button).
- **Right column** — fixed result tabs: `JSON` plus one tab per feature, derived from `FEATURES` (each feature label composed as `icon` + `tab_label`; the `JSON` tab carries an icon too). JSON shows the combined output; each feature tab renders its result (guarded by try/except), a "not enabled for this run" note if it was off, or a run prompt before the first run. An "Inputs changed since this run — click Run to refresh." note appears when the live input/toggles/language differ from the run. Rendered from `st.session_state.results`.

### Functions

- `load_model() -> tuple[nn.Module, TokenizerWrapper]` — loads model and tokenizer via `mlx_lm.load`, cached with `@st.cache_resource`.
- `truncate_to_tokens(text, tokenizer, max_tokens=MAX_INPUT_TOKENS) -> tuple[str, bool]` — truncates to a token budget; returns `(text, was_truncated)`.
- `parse_json_output(raw) -> dict | None` — returns the first JSON **object** found (tolerates surrounding prose/code fences via `JSONDecoder.raw_decode`); non-object JSON (lists, scalars) and unparseable input return `None`.
- `resolve_input(pasted, uploaded, sample) -> str` — resolves the active input by precedence (pasted > uploaded > sample); each candidate is stripped first, so a whitespace-only entry falls through.
- `language_directive(feature, language) -> str` — clause appended to the **user** turn telling the model which language to answer in. Localizes only free-text *values* (summary prose, rationale, topic labels); JSON keys and enums (e.g. the sentiment label) stay English so `parse_json_output` / `render_result` keep working. Returns `""` for `"English"`; `"Match input"` mirrors the analyzed text's language. Three wording choices are load-bearing and were each settled by an A/B against the live model (see the design decision below): the keep-English exception comes **first** and the localize requirement **last**; the JSON branch names the feature's own field (rationale / topic label) rather than "all free-text field values"; and that field name is **unquoted**. The prose branch also forbids notes about the language used, which otherwise leak in as a trailing English aside.
- `_effective_max_tokens(feature, language) -> int` — the feature's output budget, doubled for every non-English language (including "Match input") so localized JSON doesn't truncate mid-object; the base `max_tokens` for English.
- `run_feature(feature, text, model, tokenizer, language=LANGUAGE_AUTO) -> dict` — builds the feature's chat-template prompt (user turn carries the `language_directive`; system prompt stays verbatim), runs `mlx_lm.generate` with `_effective_max_tokens(feature, language)`, a greedy `make_sampler`, and — **for prose only** — a `make_logits_processors` repetition penalty; returns `{"raw": str, "parsed": dict | None}` (`parsed` only for JSON features).
- `_run_signature(input_text, enabled, language) -> tuple` — the run's identity `(input_text, toggle states, language)`; built on Run and recomputed live so the results panel can flag stale results. Single source so build/compare sides can't diverge.
- `render_result(key, result) -> None` — renders one feature's result with native components: prose for summary; a dataframe for topics (confidence as a `ProgressColumn`); `st.metric` for intent/sentiment, each with an optional rationale and a percent-formatted confidence (`_render_confidence`). The sentiment value is color-coded by enum (`_SENTIMENT_COLOR` → `:color[…]` markdown reading the theme's semantic colors, e.g. positive→green, negative→red, mixed→orange); an out-of-enum label renders uncolored. Untrusted model output is guarded — non-list `topics` falls back to a message, `st.metric` values are coerced to strings, and JSON parse failures show the raw response.

### Generation

Each feature builds `[{"role": "system", ...}, {"role": "user", ...}]`, applies the chat template with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`, and passes the resulting string to `generate(...)` with `_effective_max_tokens(feature, language)` (the base budget, doubled for every non-English output language), `make_sampler(temp=TEMP)`, and a repetition-penalty `make_logits_processors` for prose features only (`logits_processors=None` for JSON). No "thinking" mode.

### Performance

- `@st.cache_resource` caches the model; it is loaded lazily on the first Run.
- MLX handles Apple Silicon (M-series) acceleration natively.
- Inputs over `MAX_INPUT_TOKENS` (default 16384, env-configurable) are truncated before analysis.
- Fixed greedy decoding (`temp=0.0`) keeps classification output deterministic and parseable.
- No `@st.fragment`: the one expensive op (inference) is already gated behind Run and `@st.cache_resource`-cached, so isolating reruns would add complexity for no gain — and a results-panel fragment would not even see the input/toggle/language widgets it depends on (they live outside it), breaking the cross-widget staleness note. `st.session_state.results` is initialized with `st.session_state.setdefault`.

### Error Handling

Unexpected exceptions during a run — and during per-feature rendering — are shown with `st.exception()`. JSON parse failures degrade gracefully to the raw response, and `render_result` guards/coerces untrusted model output before passing it to `st.metric` / `st.dataframe`.

Both of those `except Exception` handlers carry an inline `# noqa: BLE001` naming their role (top-level run guard / untrusted model output shape). `BLE001` ("do not catch blind exception") entered ruff's **default** rule set in 0.16, and it is right about blind catches in general — but these two are deliberate UI boundaries: a Streamlit script has no frame above it to catch anything, so an escaped exception blanks the page instead of showing the user what broke. The suppression is per-line rather than a project-wide `ignore` in `pyproject.toml` **on purpose** — the rule stays live, so a third, *unintentional* blind catch added later still fails the lint gate.

## Tests

`tests/test_streamlit_app.py` — unit tests (mocked, no model download). Data-driven cases use `@pytest.mark.parametrize` (each `pytest.param` carries an `id=` so failures are self-labeling); `TestRunFeature` shares a `tokenizer` fixture:

- `TestFeatures` — `FEATURES` order, required fields (incl. each feature's `:material/…:` `icon`), every JSON feature declaring a `localized_field` that names a field its own schema declares (and prose declaring none), prose-vs-JSON outputs, `LABELS` mapping, sentence-case toggle labels (Streamlit design guidance), valid embedded JSON schemas, and the IBM-documented JSON system-prompt pattern (incl. the trailing newline)
- `TestParseJsonOutput` — plain / embedded / code-fenced JSON, first-of-multiple objects, recovery after stray braces, non-object JSON (arrays, scalars) → `None`, unparseable → `None`
- `TestResolveInput` — input precedence, whitespace stripping (incl. whitespace-only falling through to the next source), all-empty
- `TestTruncateToTokens` — short / long / boundary cases (uses `MAX_INPUT_TOKENS`), and the `add_special_tokens=False` encode flag
- `TestRunFeature` — prose vs JSON parsing, chat-template + `max_tokens` wiring, decoding-param wiring (greedy sampler always; repetition penalty prose-only), and the `language` directive landing on the user turn (and absent for English)
- `TestLanguageDirective` — `English` → empty, prose targets the language, JSON localizes values but keeps keys/enums English, and `Match input` uses the relative phrase. Further cases pin the A/B findings the design decision says not to re-litigate: each JSON feature names its **own** field (topics→`label`, intents/sentiment→`rationale`) and that name stays **unquoted**; the localize clause comes **last** (asserted with `find()`, not `rindex()`, so a dropped clause fails the assertion instead of raising `ValueError`); the prose branch suppresses language commentary and makes no claim about English at all; and the JSON branch mentions English only positively — six negative phrasings are rejected, since under "Match input" the target may itself be English. The clause-order and positive-English checks are parametrized over every JSON feature × `Match input`/Japanese/German
- `TestResolveMaxInputTokens` — env-var override, default when unset, non-integer/non-positive fallback, clamp to `MODEL_MAX_TOKENS`, and the pinned default/ceiling values
- `TestEffectiveMaxTokens` — base budget for English; doubled for **every** other entry in `LANGUAGES` (Latin-script included, and "Match input" as `LANGUAGES[0]`). Parametrized over language × feature, so all four budgets are checked against all 12 targets
- `TestRenderResult` — `st.metric` string coercion for intent/sentiment, the non-list topics guard, percent-formatted confidence, and per-enum sentiment coloring (`positive→green` … `mixed→orange`, out-of-enum → uncolored) (mocks `streamlit_app.st`)
- `TestThemeConfig` — `.streamlit/config.toml` parses, defines both light & dark modes, uses the IBM Blue primary and IBM Plex fonts, contains **only** keys Streamlit recognizes (`_flatten_theme_keys` + `streamlit.config.get_config_options()`), and has well-formed 6-digit-hex `*Color` values, plus that every hue `_SENTIMENT_COLOR` emits has a matching `*Color` key in both modes (so the sentiment metric can't silently fall back to a non-Carbon hue) — both guard the silent-degradation failure mode (Streamlit only *warns* on a bad key or color)
- `TestCIWorkflow` — `.github/workflows/ci.yml` parses, runs on an Apple Silicon (`macos-*`) runner (a Linux runner can't install the darwin-only `mlx`, so the suite wouldn't even collect), invokes all four documented gates (`ruff check`, `ruff format --check`, `ty check`, `pytest`) under `uv sync --locked`, and triggers on push to `main` and on PRs — the CI analogue of `TestThemeConfig`'s silent-degradation guard (a dropped step or swapped runner would still show green on whatever CI still ran). Parses with `pyyaml` (a dev dep); accounts for YAML 1.1 reading the bare `on:` key as boolean `True`

`tests/test_app_ui.py` — integration tests via Streamlit's `AppTest` (`streamlit.testing.v1`), driving the imperative UI block headlessly. The Run path is mocked at the **`mlx_lm` boundary** (`patch("mlx_lm.load" / "mlx_lm.generate")`) — `streamlit_app` re-execs `from mlx_lm import generate, load` on every run, so the imports bind to the mocks; an autouse fixture clears `st.cache_resource` between tests so each test's mock is used. Widgets are addressed by `key=`, not index.

- `TestInitialRender` — no model needed: Run disabled with no input, the four toggles default-on with correct labels, the Output language selectbox defaulting to "Match input", the sample picker rendering as a `segmented_control` (no `—` sentinel), and the pre-run prompt / `results is None`
- `TestUIPolish` — no model needed: the input tabs and Run button carry their Material Symbol icons, and each result tab's label is composed as `icon` + `tab_label` from `FEATURES`
- `TestRunInteraction` — Run enables once text is entered (or a built-in sample is picked); the mocked Run path populates `session_state.results`, renders the Sentiment `st.metric` and percent confidence, shows the "not enabled for this run" note for toggled-off features, flags "inputs changed" after a post-run edit (input *or* output-language change), and warns on truncated input
