# L20 Stage 2 Handoff

This document is for the AI/operator on the 8xL20 server. The goal is to finish Stage 2 explanations for the filtered Ultra-FineWeb 10k run, then build final NLA training parquets.

## Current Status

Stage 0 and Stage 1 are already complete on the source server for the 10k filtered Ultra-FineWeb run:

```text
output_dir: /tmp/nla_qwen7b_ultrafineweb_filtered_10k
base:      100000 rows, 10000 docs
av_sft:     25000 rows, 2500 docs
ar_sft:     25000 rows, 2500 docs
rl:         50000 rows, 5000 docs
d_model: 3584
layer: 20
base_model: /home/image3a/306920/model/Qwen2.5-7B-Instruct
```

Required files to copy to the L20 server:

```text
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/av_sft_raw.parquet
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/av_sft_raw.parquet.nla_meta.yaml
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/ar_sft_raw.parquet
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/ar_sft_raw.parquet.nla_meta.yaml
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/rl_raw.parquet
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/rl_raw.parquet.nla_meta.yaml
```

Keep the same directory layout on the L20 server if possible. If paths differ, update the config `output_dir` and use the copied split paths directly when running stage CLIs.

## Code Changes To Use

Use the latest pushed repo. Important additions:

- `nla.datagen.providers.OpenAIChatProvider` supports OpenAI-compatible APIs, bounded async concurrency, optional `rpm`, and fail-fast behavior.
- `nla.datagen.providers.DeepSeekProvider` supports `deepseek-v4-flash` with thinking disabled.
- `nla.datagen.stage2_api_explain` writes dropped bad responses to `{output}.bad_responses.jsonl`.
- Stage 2 is chunk-resumable. Completed `*.parquet.chunks/chunk_*.parquet` files are reused on rerun.
- `nla.datagen.benchmark_provider_concurrency` benchmarks provider throughput on real Stage 2 prompts.

Do not delete `.chunks` directories after a failure. Rerun the same command to continue.

## Option A: DeepSeek Flash Stage 2

Use this if paid API cost is acceptable and speed matters.

```bash
source /home/image3a/306920/venv/nla/bin/activate

python -m nla.datagen.run_pipeline \
  --config configs/datagen/qwen7b_ultrafineweb_filtered_10k_deepseek_flash.yaml \
  --stages 2
```

Current tested settings:

```yaml
provider_cls: nla.datagen.providers.DeepSeekProvider
model: deepseek-v4-flash
concurrency: 512
chunk_size: 512
max_tokens: 600
thinking: false
```

Benchmark on 512 real prompts: `512/512` passed after current cleaner, about `12s`.

Bad-format logs, if any:

```text
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/av_sft_explained.parquet.bad_responses.jsonl
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/ar_sft_explained.parquet.bad_responses.jsonl
```

## Option B: Local 32B Model Stage 2

Use this to avoid API cost. Start an OpenAI-compatible server with Qwen2.5-32B-Instruct.

Example vLLM server:

```bash
vllm serve /path/to/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

First benchmark real prompts:

```bash
source /home/image3a/306920/venv/nla/bin/activate

python -m nla.datagen.benchmark_provider_concurrency \
  --input /tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/av_sft_raw.parquet \
  --rows 512 \
  --concurrency 256 \
  --provider-cls nla.datagen.providers.OpenAIChatProvider \
  --provider-kwargs '{"model":"/path/to/Qwen2.5-32B-Instruct","api_base":"http://127.0.0.1:8000/v1","api_key":"dummy","max_tokens":600,"temperature":1.0,"max_retries":3,"timeout":600}'
```

If the benchmark is stable, run Stage 2 manually for AV and AR:

```bash
python -m nla.datagen.stage2_api_explain \
  --input /tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/av_sft_raw.parquet \
  --output /tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/av_sft_explained.parquet \
  --provider-cls nla.datagen.providers.OpenAIChatProvider \
  --provider-kwargs '{"model":"/path/to/Qwen2.5-32B-Instruct","api_base":"http://127.0.0.1:8000/v1","api_key":"dummy","max_tokens":600,"temperature":1.0,"concurrency":256,"max_retries":3,"timeout":600}' \
  --chunk-size 512 \
  --storage-cls nla.datagen.storage.LocalStorage

python -m nla.datagen.stage2_api_explain \
  --input /tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/ar_sft_raw.parquet \
  --output /tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/ar_sft_explained.parquet \
  --provider-cls nla.datagen.providers.OpenAIChatProvider \
  --provider-kwargs '{"model":"/path/to/Qwen2.5-32B-Instruct","api_base":"http://127.0.0.1:8000/v1","api_key":"dummy","max_tokens":600,"temperature":1.0,"concurrency":256,"max_retries":3,"timeout":600}' \
  --chunk-size 512 \
  --storage-cls nla.datagen.storage.LocalStorage
```

Tune `concurrency` from benchmark results. Do not start with 500k calls; benchmark 512 or 2048 prompts first.

## Build Final Training Parquets

After Stage 2 succeeds for both AV and AR, run Stage 3 and shuffle:

```bash
python -m nla.datagen.run_pipeline \
  --config configs/datagen/qwen7b_ultrafineweb_filtered_10k_deepseek_flash.yaml \
  --stages 3,shuffle
```

If using local 32B manually for Stage 2, this same command is still fine because Stage 3 consumes only the explained parquet files.

Final files to return to the training server:

```text
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/av_sft_shuf.parquet
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/av_sft_shuf.parquet.nla_meta.yaml
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/ar_sft_shuf.parquet
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/ar_sft_shuf.parquet.nla_meta.yaml
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/rl_shuf.parquet
/tmp/nla_qwen7b_ultrafineweb_filtered_10k/rl_shuf.parquet.nla_meta.yaml
```

Stage 3 needs the Qwen2.5-7B tokenizer path recorded in sidecars. Ensure `/home/image3a/306920/model/Qwen2.5-7B-Instruct` exists on the L20 server, or create a symlink to the local copy.

## Monitoring

Stage 2 progress:

```bash
watch -n 30 'find /tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits -path "*.parquet.chunks/chunk_*.parquet" | wc -l'
```

For DeepSeek flash with `chunk_size=512`, expected total chunks:

```text
av_sft: 49
ar_sft: 49
total: 98
```

If any rows are dropped, inspect:

```bash
wc -l /tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/*bad_responses.jsonl
head -1 /tmp/nla_qwen7b_ultrafineweb_filtered_10k/splits/*bad_responses.jsonl
```

## Quality Notes

MiMo followed formatting more naturally but is limited to about `100 RPM`, making it too slow for 100k. DeepSeek flash is much faster but often writes features as `First/Second/Third` in one paragraph. The Stage 2 cleaner now handles that and logs any remaining bad responses.

Do not change data-gen invariants: raw activation vectors only (`norm="none"`), document-level split by `doc_id`, Stage-0 `_MIN_POSITION = 50`, and preserve sidecars.
