# Granite Text Intelligence

**Granite Text Intelligence** analyzes text with IBM's [granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) running locally on Apple Silicon via `mlx-lm` (MLX acceleration). It's a single-shot playground: provide text, choose which analyses to run, and get **summarization, topic detection, intent recognition, and sentiment** back — all powered by prompting one Granite model. Results can be returned in the input's language or any of Granite's 12 supported languages.

Requires an Apple Silicon (M-series) Mac with ~24 GB+ of unified memory (32 GB recommended) — the 8B model uses ~16.8 GB in bf16. On lower-memory Macs, set `MODEL_NAME` in `streamlit_app.py` to a 4-bit quant such as `mlx-community/granite-4.1-8b-4bit` (~5.2 GB) to cut memory roughly 3× for a small quality cost.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The model (~16.8 GB, bf16) downloads automatically on first run.

### Hugging Face token (optional)

The Granite model is public, so no token is required. Without one, the Hugging Face Hub logs a `You are sending unauthenticated requests to the HF Hub` warning and applies lower rate limits and slower downloads.

To authenticate, copy the template and set a token with **read** scope ([create one](https://huggingface.co/settings/tokens)):

```bash
cp .env.example .env
# then edit .env and set HF_TOKEN=hf_...
```

`.env` is gitignored and loaded automatically via `python-dotenv`.

**Deployment:** set `HF_TOKEN` as an environment variable in your platform's secrets instead of shipping `.env`. `load_dotenv()` does not override real env vars and no-ops when no `.env` is present, so the same code works locally and in production.

### Input length (optional)

Inputs over `MAX_INPUT_TOKENS` tokens (default `16384`, max `131072`) are truncated before analysis. On a higher-memory Mac you can raise it for longer documents, but each extra token adds ~160 KB of KV cache and slows processing. Set it like `HF_TOKEN` — in `.env` or as an environment variable:

```bash
MAX_INPUT_TOKENS=32768
```

## Usage

1. Provide text via one of the **Text**, **Upload**, or **Sample** tabs (when more than one has content, precedence is Text > Upload > Sample).
2. (Optional) Pick an **Output language** — "Match input" (default) mirrors the input's language, or choose one of Granite's 12 supported languages.
3. Toggle the analyses you want: **Summarization**, **Topic Detection**, **Intent Recognition**, **Sentiment**.
4. Click **Run**.
5. Read the results in the per-feature tabs, plus a combined **JSON** tab.

## Features

- **Four analyses** — summarization (prose), plus topic detection, intent recognition, and sentiment (structured JSON), each a task-specific Granite prompt
- **Three input sources** — paste text, upload a `.txt`/`.md` file, or pick a built-in sample
- **Per-feature toggles** — run exactly the analyses you want; each description lives in the toggle's tooltip
- **Multilingual output** — return results in the input's language or any of Granite's 12 supported languages
- **Configurable input length** — `MAX_INPUT_TOKENS` env var (default 16384) caps how much text is analyzed before truncation
- **Tabbed results** — readable per-feature views plus a combined JSON view
- **Local and private** — runs entirely on-device via MLX; no text leaves your Mac

## Development

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check        # typecheck
uv run pytest          # test
```

Configuration is in `pyproject.toml`.
