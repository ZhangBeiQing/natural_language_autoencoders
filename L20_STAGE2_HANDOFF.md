# L20 Training Handoff

This handoff is for the AI/operator on the 8xL20 training server. Stage 2 API
explanation generation has already been completed with DeepSeek; do not build a
local inference service or call DeepSeek again unless explicitly asked.

## Goal

Run the full NLA training path once on the 10k filtered Ultra-FineWeb dataset:

1. Critic SFT on AR data.
2. Actor SFT on AV data.
3. RL using the actor and critic checkpoints.

This is a 10k smoke/midscale run to validate the training stack, not the final
100k production dataset.

## Required Data

Copy these six files to the training server. The three sidecars are required;
the parquet files alone are not enough.

```text
/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k/av_sft_shuf.parquet
/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k/av_sft_shuf.parquet.nla_meta.yaml
/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k/ar_sft_shuf.parquet
/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k/ar_sft_shuf.parquet.nla_meta.yaml
/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k/rl_shuf.parquet
/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k/rl_shuf.parquet.nla_meta.yaml
```

Recommended destination:

```bash
export DATA_DIR=/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k
```

Expected row counts:

```text
av_sft_shuf.parquet: 24975 rows
ar_sft_shuf.parquet: 24979 rows
rl_shuf.parquet:     50000 rows
```

Optional audit files are under `splits/*.bad_responses.jsonl`; they are not
needed for training.

## Code And Model Requirements

Use repo commit `90b3c3c` or newer. Activate the project environment:

```bash
source /home/image3a/306920/venv/nla/bin/activate
```

The sidecars were generated against:

```bash
export INSTRUCT_MODEL=/home/image3a/306920/model/Qwen2.5-7B-Instruct
```

If the model lives elsewhere on the L20 server, either set `INSTRUCT_MODEL` to
that path or create a symlink at the path above. Do not edit the parquet files or
sidecars. The Qwen injection token is `㈎` with token id `149705`.

## Sanity Check

Run this before training:

```bash
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

data = Path("/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k")
for name in ["av_sft_shuf.parquet", "ar_sft_shuf.parquet", "rl_shuf.parquet"]:
    p = data / name
    sidecar = Path(str(p) + ".nla_meta.yaml")
    print(name, pq.ParquetFile(p).metadata.num_rows, "sidecar=", sidecar.exists())
PY
```

Also run a quick syntax check after pulling the latest code:

```bash
python -m compileall nla nla_inference.py
```

## Training Commands

Set shared paths:

```bash
export DATA_DIR=/home/image3a/306920/dataset/nla_qwen7b_ultrafineweb_filtered_10k
export INSTRUCT_MODEL=/home/image3a/306920/model/Qwen2.5-7B-Instruct
export AV_SFT_PARQUET=$DATA_DIR/av_sft_shuf.parquet
export AR_SFT_PARQUET=$DATA_DIR/ar_sft_shuf.parquet
export RL_PARQUET=$DATA_DIR/rl_shuf.parquet
```

Prepare the truncated critic init checkpoint. `--num-layers 20` matches the
activation extraction layer in the dataset.

```bash
export CRITIC_INIT_CKPT=/path/to/outputs/critic_init_qwen7b_layer20

python -m nla.scripts.prepare_critic_checkpoint \
  --base-model "$INSTRUCT_MODEL" \
  --num-layers 20 \
  --dataset-sidecar "$AR_SFT_PARQUET" \
  --output "$CRITIC_INIT_CKPT" \
  --megatron-compat
```

Run Critic SFT:

```bash
export SAVE_DIR=/path/to/outputs/critic_sft_10k
bash configs/critic_sft.sh
```

Run Actor SFT:

```bash
export SAVE_DIR=/path/to/outputs/actor_sft_10k
export INJ_SCALE=sqrt_d_model
bash configs/actor_sft.sh
```

Run RL after choosing the actor and critic checkpoints from the SFT outputs:

```bash
export ACTOR_SFT_CKPT=/path/to/outputs/actor_sft_10k/iter_XXXXXXX
export CRITIC_SL_CKPT=/path/to/outputs/critic_sft_10k/iter_XXXXXXX/hf
export RUN_DIR=/path/to/outputs/rl_10k
bash configs/rl.sh
```

If `INJ_SCALE=sqrt_d_model` is not accepted by the current training code, use
`INJ_SCALE=59.866` because `sqrt(3584) ~= 59.866`.

## Invariants

Keep these unchanged while debugging:

- Data vectors are raw activations; normalization is represented by sidecar
  metadata and training-time scaling.
- Stage 1 was split at document level by `doc_id`.
- `d_model=3584`, `layer=20`, and `cp_size` must stay compatible with Qwen2.5-7B.
- RL must keep SGLang radix cache disabled because NLA injects different raw
  activations behind the same marker token.
- Do not regenerate data or filter rows on the training server.

## Report Back

After the run, report:

- Row-count sanity check output.
- Critic SFT first-loss and latest-loss lines.
- Actor SFT first-loss and latest-loss lines.
- RL startup result, first completed steps, and checkpoint paths.
- Any tokenizer or `nla_meta.yaml` assertion exactly as printed.
