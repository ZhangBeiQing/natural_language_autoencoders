# NLA Data Generation — External Server Instructions

## Context

We're running the NLA (Natural Language Autoencoder) baseline experiment. The internal server cannot access HuggingFace or Anthropic API, so Stage 0-2 of data generation must run on this external server. Results will be transferred back.

## Prerequisites

- Python 3.10+
- GPU with >= 16GB VRAM (Qwen2.5-7B needs ~15GB in bf16)
- `ANTHROPIC_API_KEY` environment variable set
- Model already at: `/media/image3a/f/model/Qwen2.5-7B-Instruct`

## Step 1: Clone and install

```bash
cd /media/image3a/f
git clone https://github.com/anthropics/natural-language-autoencoders.git nla-repo
cd nla-repo
pip install -e .
```

If the GitHub repo is not accessible, try the paper's repo or check if the code is available at an alternative location. The core code needed is in the `nla/datagen/` directory.

## Step 2: Run Stage 0 (activation extraction) + Stage 1 (split)

This extracts hidden-state activations from Qwen2.5-7B at layer 20 over 1000 FineWeb documents, then splits into AV-SFT / AR-SFT / RL subsets.

```bash
python -m nla.datagen.run_pipeline \
    --config configs/datagen/qwen7b_fineweb_10k.yaml \
    --stages 0,1 \
    --override "base_model=/media/image3a/f/model/Qwen2.5-7B-Instruct"
```

**Important notes:**
- This will download a small portion of FineWeb `sample-10BT` dataset from HuggingFace (~1GB, not the full TB-scale dataset). The `datasets` library downloads only the parquet shards it needs for 1000 documents.
- If `multigpu: true` causes issues (the multi-GPU script launches multiple processes), edit the config to set `multigpu: false`, or run stage0 directly:

```bash
# Single-GPU fallback
python -m nla.datagen.stage0_extract \
    --base-model /media/image3a/f/model/Qwen2.5-7B-Instruct \
    --corpus HuggingFaceFW/fineweb \
    --corpus-config sample-10BT \
    --corpus-split train \
    --corpus-start 0 --corpus-length 1000 \
    --layer-index 20 \
    --positions-per-doc 10 \
    --chunk-size 256 \
    --seed 42 \
    --output /tmp/nla_qwen7b_fineweb_10k/base.parquet \
    --extractor-kwargs '{"batch_size": 16, "max_length": 2048}'

# Then Stage 1
python -m nla.datagen.stage1_split \
    --base /tmp/nla_qwen7b_fineweb_10k/base.parquet \
    --av-sft-frac 0.25 --ar-sft-frac 0.25 --rl-frac 0.50 \
    --seed 42 \
    --output-dir /tmp/nla_qwen7b_fineweb_10k/splits
```

Stage 0 takes ~30-60 minutes on a single GPU (depends on GPU speed). Stage 1 is instant.

## Step 3: Run Stage 2 (API explanation generation)

This calls Claude Haiku to generate natural-language explanations for the source text. Requires `ANTHROPIC_API_KEY`.

```bash
export ANTHROPIC_API_KEY=your_key_here

python -m nla.datagen.run_pipeline \
    --config configs/datagen/qwen7b_fineweb_10k.yaml \
    --stages 2 \
    --override "base_model=/media/image3a/f/model/Qwen2.5-7B-Instruct"
```

Or run directly:

```bash
export ANTHROPIC_API_KEY=your_key_here

# AV-SFT explanations
python -m nla.datagen.stage2_api_explain \
    --input /tmp/nla_qwen7b_fineweb_10k/splits/av_sft_raw.parquet \
    --output /tmp/nla_qwen7b_fineweb_10k/splits/av_sft_explained.parquet \
    --provider-cls nla.datagen.providers.AnthropicProvider \
    --provider-kwargs '{"model": "claude-haiku-4-5-20251001", "max_tokens": 300, "concurrency": 400}' \
    --chunk-size 512

# AR-SFT explanations
python -m nla.datagen.stage2_api_explain \
    --input /tmp/nla_qwen7b_fineweb_10k/splits/ar_sft_raw.parquet \
    --output /tmp/nla_qwen7b_fineweb_10k/splits/ar_sft_explained.parquet \
    --provider-cls nla.datagen.providers.AnthropicProvider \
    --provider-kwargs '{"model": "claude-haiku-4-5-20251001", "max_tokens": 300, "concurrency": 400}' \
    --chunk-size 512
```

This costs ~$2-5 in API calls (2500 + 2500 = 5000 completions at ~200 input tokens each). Takes ~5-15 minutes depending on API rate limits.

## Step 4: Package results for transfer

```bash
cd /tmp
tar czf nla_qwen7b_fineweb_10k_stage012.tar.gz nla_qwen7b_fineweb_10k/
```

Expected output structure:

```
/tmp/nla_qwen7b_fineweb_10k/
├── base.parquet                        # ~50-200MB, raw activation vectors
├── base.parquet.nla_meta.yaml          # sidecar metadata
├── splits/
│   ├── av_sft_raw.parquet              # AV-SFT subset (no explanations yet)
│   ├── av_sft_raw.parquet.nla_meta.yaml
│   ├── ar_sft_raw.parquet              # AR-SFT subset
│   ├── ar_sft_raw.parquet.nla_meta.yaml
│   ├── rl_raw.parquet                  # RL subset
│   ├── rl_raw.parquet.nla_meta.yaml
│   ├── av_sft_explained.parquet        # AV-SFT with api_explanation column
│   ├── av_sft_explained.parquet.nla_meta.yaml
│   ├── ar_sft_explained.parquet        # AR-SFT with api_explanation column
│   └── ar_sft_explained.parquet.nla_meta.yaml
```

**Transfer this tar.gz to the internal server at `/tmp/`.** We will then run Stage 3 (build training parquets) + shuffle + training on the internal server.

## Troubleshooting

- **FineWeb download fails**: The `datasets` library may have trouble with large datasets. Try `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads, or pre-download the parquet files.
- **OOM during Stage 0**: Reduce `batch_size` in `extractor_kwargs` (e.g., from 16 to 4) and/or reduce `max_length` (e.g., from 2048 to 1024).
- **API rate limits**: Reduce `concurrency` in provider_kwargs (e.g., from 400 to 50) and increase `max_retries`.
- **No ANTHROPIC_API_KEY**: You can write a custom provider that uses a local model instead. See `nla/datagen/providers.py` for the `CompletionProvider` interface — just implement `complete(prompts) -> list[str|None]`.
