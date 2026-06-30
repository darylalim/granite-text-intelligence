# Granite Text Intelligence

[![CI](https://github.com/darylalim/granite-text-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/granite-text-intelligence/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/darylalim/granite-text-intelligence)](https://github.com/darylalim/granite-text-intelligence/releases/latest)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Granite Text Intelligence** is a Streamlit application for **summarization, topic, intent, and sentiment analysis** using IBM's [granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) on Apple Silicon with [MLX](https://github.com/ml-explore/mlx) (Apple's on-device ML framework, via `mlx-lm`), running locally. It's a single-shot playground: provide text, choose which analyses to run, and get the results back — all powered by prompting one Granite model. Results can be returned in the input's language or any of Granite's 12 supported languages.

Requires an Apple Silicon (M-series) Mac with ~24 GB+ of unified memory (32 GB recommended) — the 8B model uses ~16.8 GB in bf16 (bfloat16). On lower-memory Macs, set `MODEL_NAME` in `streamlit_app.py` to a 4-bit quant such as `mlx-community/granite-4.1-8b-4bit` (~5.2 GB) to cut memory roughly 3× for a small quality cost.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), which provisions Python 3.12 (per `.python-version`) and all dependencies for you.

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The model (~16.8 GB, bf16) downloads automatically the first time you click **Run** (you'll see a "Loading model…" spinner) and is cached for later runs.

### Troubleshooting

- **First Run is slow.** The initial click loads ~16.8 GB into unified memory; the "Loading model…" and per-feature spinners mean it's working, not hung.
- **Out of memory?** Switch `MODEL_NAME` to the 4-bit quant noted above (`mlx-community/granite-4.1-8b-4bit`), which needs ~5.2 GB.
- **Interrupted download?** Re-run — downloads resume from the Hugging Face cache rather than starting over.

## Usage

1. Provide text via one of the **Text**, **Upload**, or **Sample** tabs (when more than one has content, precedence is Text > Upload > Sample).
2. (Optional) Pick an **Output language** — "Match input" (default) mirrors the input's language, or choose one of Granite's 12 supported languages: English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, Chinese.
3. Toggle the analyses you want: **Summarization**, **Topic detection**, **Intent recognition**, **Sentiment**.
4. Click **Run**.
5. Read the results in the per-feature tabs, plus a combined **JSON** tab.

## Configuration

Both settings below are optional. Set them like any environment variable — in `.env` (gitignored, loaded automatically via `python-dotenv`) or as a real environment variable.

### Hugging Face token

The Granite model is public, so no token is required. Without one, the Hugging Face Hub logs a `You are sending unauthenticated requests to the HF Hub` warning and applies lower rate limits and slower downloads.

To authenticate, copy the template and set a token with **read** scope ([create one](https://huggingface.co/settings/tokens)):

```bash
cp .env.example .env
# then edit .env and set HF_TOKEN=hf_...
```

**Deployment:** set `HF_TOKEN` as an environment variable in your platform's secrets instead of shipping `.env`. `load_dotenv()` does not override real env vars and no-ops when no `.env` is present, so the same code works locally and in production.

### Input length

Inputs over `MAX_INPUT_TOKENS` tokens (default `16384`, max `131072`) are truncated before analysis. On a higher-memory Mac you can raise it for longer documents, but each extra token adds ~160 KB of KV cache and slows processing. Add it to `.env`, or pass it inline for a single run:

```bash
MAX_INPUT_TOKENS=32768 uv run streamlit run streamlit_app.py
```

## Features

- **Four analyses** — summarization (prose), plus topic detection, intent recognition, and sentiment (structured JSON), each a task-specific Granite prompt
- **Per-feature toggles** — run exactly the analyses you want; each description lives in the toggle's tooltip
- **Tabbed results** — readable per-feature views plus a combined JSON view
- **IBM Carbon-inspired UI** — IBM Plex fonts and an IBM Blue accent, with light/dark mode and Material Symbol icons throughout
- **Local and private** — runs entirely on-device via MLX; no text leaves your Mac

## Development

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check        # typecheck
uv run pytest          # test
```

These four checks also run in CI on every push to `main` and pull request (see the badge above), on an Apple Silicon (macOS) runner — CI is macOS-only because the darwin-only `mlx` can't install on Linux. Tooling configuration (ruff, pytest, ty) lives in `pyproject.toml`.

Contributions are welcome — open an issue or PR. Before submitting, run the four checks above (or rely on the project's Claude Code hooks, which run them on edit and on stop) so CI stays green.

## License

This project's code is released under the [Apache License 2.0](LICENSE). The IBM Granite model it loads is distributed separately under [its own Apache 2.0 license](https://huggingface.co/ibm-granite/granite-4.1-8b) and is downloaded at runtime, not included in this repository.
