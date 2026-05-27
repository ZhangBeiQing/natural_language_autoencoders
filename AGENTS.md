# Repository Guidelines

## Project Structure & Module Organization

`nla/` is the core Python package. Important subpackages include `nla/datagen/` for the staged activation-to-parquet pipeline, `nla/megatron/` for Megatron training integration, `nla/rollout/` for rollout helpers, and `nla/miles_patches/` for documented upstream Miles patches. `nla_inference.py` is the standalone inference client. `configs/` contains training shell scripts and datagen YAMLs, `docs/` contains setup/design/inference notes, `tools/` contains checkpoint conversion utilities, `patches/` contains SGLang patches, `release/` contains Hub release helpers, and `examples/` contains worked transcripts.

## Build, Test, and Development Commands

Use the project virtual environment before running Python commands:

```bash
source /home/image3a/306920/venv/nla/bin/activate
```

Install locally with:

```bash
pip install -e .
```

Run a quick syntax check before submitting:

```bash
python -m compileall nla nla_inference.py
```

Run inference against an SGLang server:

```bash
python nla_inference.py kitft/nla-qwen2.5-7b-L20-av --sglang-url http://localhost:30000
```

Run data generation or training with the documented configs:

```bash
python -m nla.datagen.run_pipeline --config configs/datagen/quick_test_10docs.yaml
bash configs/critic_sft.sh
bash configs/actor_sft.sh
bash configs/rl.sh
```

## Coding Style & Naming Conventions

Use Python 3.10+ and keep code compatible with the dependencies in `pyproject.toml`. The Ruff line length is 119 characters. Use 4-space indentation, typed dataclasses or structured config objects where already used, and `argparse` for CLIs under `nla/`. Prefer import-path extension hooks such as `--storage-cls` and `--provider-cls` over hardcoded backends. Do not add private/internal dependencies.

## Testing Guidelines

There is no dedicated test suite in this checkout. For changes, run `python -m compileall` and the smallest relevant smoke path, such as `quick_test_10docs.yaml` for datagen changes or `nla_inference.py` for serving changes. If adding tests, keep them near the affected module, name files `test_*.py`, and avoid heavyweight model downloads unless explicitly marked.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout, so use standard concise, imperative commit subjects such as `Fix datagen sidecar validation`. Pull requests should describe the behavior change, list commands run, call out required GPUs/API keys, and link related issues or experiments. Include config diffs when training behavior changes.

## Agent-Specific Invariants

Do not edit upstream Miles code; extend via `NLAFSDPActor` and function-pointer arguments. Data generation must preserve raw vectors with `norm="none"`; normalization belongs at injection and loss time via sidecar scales. Stage-1 splitting is document-level by `doc_id`. Keep Stage-0 `_MIN_POSITION = 50`. Critic extraction is suffix-anchored at `tokens[-1]`. Preserve per-doc keyed RNG reproducibility. Injection hooks must scan token IDs inside the hook, not use precomputed positions. Keep `cp_size == 1`. Treat `nla_meta.yaml` as the contract for token IDs, prompt templates, scales, and `d_model`.

## Network & Proxy

This server cannot access foreign servers directly. Run `proxy_on` from `~/.bashrc` before external network access. Without proxy, the internal pip mirror (`mirrors.dahuatech.com`) and HuggingFace mirror (`hf-mirror.com` via `HF_ENDPOINT`) work for many packages and models, but HuggingFace dataset file downloads can fail. With proxy, switch to official sources:

```bash
proxy_on
export HF_ENDPOINT=https://huggingface.co
pip install <pkg> -i https://pypi.org/simple/
```

Use proxy plus official sources for `datasets.load_dataset()`, `lm_eval`, `pip install git+https://...`, and first-time HuggingFace dataset/model downloads. If a download or install hangs or repeatedly fails, stop and report the failure instead of leaving a terminal blocked.

## Local Datagen Defaults

For the external-server Qwen baseline, use the local model at `/home/image3a/306920/model/Qwen2.5-7B-Instruct`. Stage 2 currently uses `nla.datagen.providers.MiMoProvider` with `mimo-v2.5-pro` through the OpenAI-compatible endpoint. Read credentials from environment variables only: `MIMO_API_BASE` and `MIMO_API_KEY`. Never commit API keys or write them into config files.
