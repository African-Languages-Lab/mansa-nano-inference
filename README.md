# mansa-nano — inference

A minimal, inference-only package for **mansa-nano**, a small English-centric chat model
trained by Sheriff Issaka and the African Languages Lab as part of a ~$100 training
experiment. This repo contains only what is needed to *serve* the model — no training,
data, or evaluation code. Weights are pulled from the Hugging Face Hub on startup.

## Quickstart

```bash
# 1. (recommended) create a fresh environment
python -m venv .venv && source .venv/bin/activate

# 2. install deps (see requirements.txt for a hardware-specific torch build)
pip install -r requirements.txt

# 3. serve the web UI + API (auto-downloads weights from Hugging Face)
python -m scripts.serve
```

Then open http://localhost:8000 in your browser.

Runs on CPU, a single GPU (CUDA), or Apple Silicon (MPS) — the device is autodetected.
bf16 weights are automatically cast to fp32 on CPU/MPS.

## Command-line chat

```bash
python -m scripts.chat_cli -p "What is the capital of France?"
```

## API

`POST /chat/completions` (streaming, Server-Sent Events):

```bash
curl -N http://localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

Other endpoints: `GET /` (web UI), `GET /logo.svg`, `GET /health`, `GET /stats`.

## Configuration

Common flags (both `scripts/serve.py` and `scripts/chat_cli.py`):

| Flag | Default | Description |
| --- | --- | --- |
| `--hf-repo` | `African-Languages-Lab/mansa-nano` | HF repo hosting the weights |
| `--local-dir` | _(none)_ | Load from a local checkpoint dir, skip HF download |
| `--step` | latest | Checkpoint step to load |
| `-t/--temperature` | `0.8` (serve) / `0.6` (cli) | Sampling temperature |
| `-k/--top-k` | `50` | Top-k sampling |
| `-m/--max-tokens` | `512` | Max tokens per response |
| `--system-prompt` | persona | Persona prepended to first user turn (`""` to disable) |
| `--device-type` | autodetect | `cuda` \| `cpu` \| `mps` |

`scripts/serve.py` additionally supports `-n/--num-gpus` (CUDA only), `-p/--port`, `--host`.

If the HF repo is **private**, set `HF_TOKEN` (or run `huggingface-cli login`) on the host.

## Publishing weights (maintainers)

Upload the three inference files (`model_<step>.pt`, `meta_<step>.json`, `tokenizer.pkl`)
to the Hub:

```bash
python export_to_hf.py --src /path/to/run_archive/<run_id> --repo-id African-Languages-Lab/mansa-nano
```
