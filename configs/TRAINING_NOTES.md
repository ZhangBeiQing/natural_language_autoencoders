# Training Notes (Qwen2.5-7B case study)

> **These are the settings we used, not settings we claim are optimal.** We did
> not sweep batch size, learning rate, or GRPO group size for RL, and do not
> regard the configurations below as near-optimal. They are a working point
> that produced the released checkpoints; treat the LR scans and memory notes
> as a starting place, not a recommendation.

The Qwen2.5-7B run was the most thoroughly profiled. The engineering lessons
here (FA2 + m16 without gradient checkpointing, the response-length cap, the
`list→np.asarray` GC fix) carry over to the other three models. We did **not**
re-sweep LR / batch size per model — the shipped `configs/*.sh` are the Qwen
settings with only light adjustment, so there is likely headroom from a
per-model tune.

Measured on 2× H100-80GB. Data: 100k UltraFineWeb documents × 5 vectors =
500k pairs, split evenly between AV and AR SFT.

## Actor SFT (full 28-layer model)

```bash
--actor-num-gpus-per-node 2
--rollout-batch-size 256 --global-batch-size 256
--micro-batch-size 16           # grad_accum=8; m32+ OOMs without grad ckpt
--num-rollout 1000              # ≈1 epoch on 250k rows
--lr 2e-5 --min-lr 2e-6         # see LR scan — 5e-5 converges faster on TRAIN loss but
                                #   real-vs-rand gap (~0.21) is the signal, not train loss.
                                #   Sticking with 2e-5 where the gap is verified stable.
--lr-warmup-iters 50            # 5% of num-rollout
--lr-decay-style cosine
--attn-implementation flash_attention_2
                                # NO --gradient-checkpointing — m16 fits without it, 36% faster
--save-interval 500             # checkpoint at halfway + final
--nla-injection-scale 150
--loss-mask-type qwen           # masks out prompt, loss only on response tokens
```

**Step time**: **4.97s** with all fixes (was 14.19s — **2.85× faster**). ~44% MFU. 94% GPU util (wait ~6%).
**Peak memory**: 67-80GB / 82GB (post-clear drops to ~33GB).
**Loss trajectory**: 4.4 (step 0) → 2.9 (warmup end) → 1.5 (step 300) → still dropping.
**vs random baseline** (shuffled activation vectors): ~0.2 lower loss by step 300 — real signal being learned.

## Critic SL (20-layer truncated model)

```bash
--actor-num-gpus-per-node 2
--rollout-batch-size 256 --global-batch-size 256
--micro-batch-size 64
--num-rollout 1000
--lr 2e-5 --min-lr 2e-6         # matched to actor — worked well
--lr-warmup-iters 50
--lr-decay-style cosine
--attn-implementation sdpa
                                # NO gradient checkpointing needed (fits at ~67GB)
--save-interval 500
```

**Step time**: ~3s. ~31% MFU.
**Peak memory**: ~67GB / 82GB.
**Loss trajectory**: 1.61 (step 0, identity init) → 1.08 (step 23) → 0.72 (step 380) → **0.586** final.
**Predict-the-mean baseline**: 0.938 (the critic's achievable loss with constant pred = normalize(μ),
since the loss normalizes pred too). **FVE = 1 − 0.586/0.938 = 37.5%**. Note: critic_rand (shuffled
targets) got 0.922 ≈ baseline — can only do constant-pred with no real signal, as expected. Computed
automatically at startup (`nla.schema.load_predict_mean_baselines`) and logged per-step as `train/fve`.

### ⚠️ Critical: identity-init `value_head`

`prepare_critic_checkpoint` must set `value_head.weight = torch.eye(d)`. PyTorch's default
`nn.Linear` init (kaiming_uniform) scales the backbone's output norm by ~1/√3, making step-0
`pred_norm ≈ 48` when `backbone_norm ≈ 83`. With identity init, `pred_norm ≈ backbone_norm` at
step 0 and initial loss drops from 1.94 → 1.61 (~17% better starting direction match).

## Memory notes — why actor needs grad checkpointing

Actor OOMs without grad ckpt at **any** micro_bsz ≥ 32. Confirmed failures:
- SDPA no-ckpt m64: 74.9GB OOM (forward)
- SDPA no-ckpt m32: 73.3GB OOM (backward — FSDP param gather + activations)
- FA2 no-ckpt m64/48/32: all OOM (FA2's O(T²) score-matrix saving is ~3MB at seq=247; negligible vs ~35GB MLP intermediates)
- FA2+ckpt m128: OOM (even with ckpt, m128 activations don't fit)

**Why actor ≠ critic** (both at m64, only actor needs ckpt):
| | actor | critic |
|---|---|---|
| seq length | 247 (prompt 125 + resp 122) | 126 |
| layers | 28 | 20 |
| output head | lm_head (d×vocab = 545M params, 4.8GB logits) | value_head (d×d = 13M params) |

All three factors compound. The lm_head forward alone is ~17 TFLOPs — equivalent to ~7
transformer layers worth of linear ops.

## Config sweep results — use **flash_attention_2 + NO grad ckpt + micro=16**

| config | step_time | grad_accum | vs baseline |
|---|---|---|---|
| **FA2 + no-ckpt + m16** | **9.05s** | 8 | **← best — 36% faster** |
| FA2 + ckpt + m64 | 12.83s | 2 | −10% |
| sdpa + ckpt + m64 | 14.19s | 2 | baseline |
| FA2 no-ckpt m64/48/32 | OOM | — | activations too large |
| FA2 + ckpt + m96/128/160 | OOM | — | peak ~67GB at m64 is already ceiling |
| sdpa no-ckpt m24 | 479s (!) | 5.33 (!) | non-integer grad_accum → pathological |

**Why m16 wins**: FLOP-equivalence — 8 microbatches × (fwd+bwd) = 2 microbatches × (fwd+recompute+bwd).
The extra FSDP gather/scatter overhead (6 more rounds ≈ ~1s on NVLink) is less than the saved
recompute cost (~4s). Memory: m16 activations fit where m32 OOMed at 73GB.

**Recommended**: `--attn-implementation flash_attention_2 --micro-batch-size 16` (NO `--gradient-checkpointing`).

### Critical data pipeline fix — `list` → `np.asarray` in NLADataSource

`data_source.py` previously stored `activation_vector` as `list(row[...])` — 250k rows × 3584
floats = **896M list slots** in the RolloutManager heap. Python's cyclic GC scans ALL slots
every gen2 collection (triggered by tokenizer allocation churn) → **2.6s stall per GC fire**,
~1.4 fires per rollout → 3.6s of the 4.2s `train_wait_time`.

Fix: `np.asarray(dtype=float32)`. Numpy arrays are GC-atomic. `torch.tensor`
consumers unchanged. **Measured step_time 9.5s → 8.3s** (isolated benchmark predicted ~6s but
real training has more heap). Also: dataset init 2× faster, RSS −4GB. Remaining ~3.3s train_wait
is Ray serialize + convert_samples_to_train_data — next target.

## LR scan (60 steps, warmup=10)

| LR | step-60 loss |
|---|---|
| 2e-5 | 1.857 |
| **5e-5** | **1.680** ← best |
| 1e-4 | 1.704 |
| 2e-4 | 1.718 |

**200-step confirmation**: lr=5e-5 late → loss **1.510**, matching lr=2e-5 @ step **500**.
**2.5× speedup** in steps-to-same-loss on *train loss alone*.

**But train loss isn't the target — real-vs-rand gap is.** The main 1000-step run at lr=2e-5
shows a stable ~0.21 gap (real 1.50 vs rand 1.71). We haven't verified 5e-5 preserves this gap
(scan only ran real). **Sticking with 2e-5** where the gap is known good. Revisit if RL needs
faster critic retraining per rollout.

### LR scaling when changing batch size

Reference: **lr=2e-5 at batch=256** (our production setting). Scale LR by √(batch/256):

| batch | recommended LR |
|---|---|
| 128 | 1.4e-5 |
| 256 | 2e-5 (reference) |
| 512 | 2.8e-5 |
| 1024 | 4e-5 |

This is the sqrt rule for Adam-family optimizers. `configs/actor_sft.sh` prints a warning at launch if
your LR is >2× off the sqrt-scaled recommendation.

## RL config — the settings we used (see caveat at top)

```bash
--lr 1e-5 --critic-lr 5e-5           # LR scan winner (9 combos, 30 steps each)
--rollout-max-response-len 150       # stops length drift at the source. At high LR the critic
                                     #   rewards verbosity → resp_len drifts 123→165+ otherwise.
                                     #   With cap=150, m16 works fine (seq~275, no OOM).
--micro-batch-size 16                # m16 is fine with resp_len capped at 150
--rollout-batch-size 64 --n-samples-per-prompt 8  # 512 gens/rollout, GRPO over 8
--attn-implementation flash_attention_2
# NO --gradient-checkpointing — causes NCCL deadlock in update_weights()
#   (FSDP full-param gather behaves differently, broadcast hangs, 10min watchdog SIGABRT)
```

**LR scan results** (30 steps each, fve_nrm @ final step):
| actor_lr | critic_lr | fve_nrm | notes |
|---|---|---|---|
| 5e-7 | 5e-7 | 0.175 | too slow |
| 1e-6 | 1e-6 | 0.188 | |
| 2e-6 | 2e-6 | 0.212 | |
| 5e-6 | 5e-6 | 0.295 | |
| 1e-5 | 1e-5 | 0.377 | phase1 winner |
| 1e-5 | 2e-5 | 0.429 | |
| **1e-5** | **5e-5** | **0.483** | **← winner** |
| 1e-5 | 1e-4 | 0.453@s26 | OOM (resp_len=187), trend reversed |
| *1e-6* | *2e-5* | *0.265* | *original baseline (mismatched)* |

(This Qwen-only scan informed our choice; we did not systematically sweep across models.)

**Lesson**: actor LR was the original bottleneck. 10× higher actor LR + 5× critic works.
Higher critic LR makes critic reward longer responses → actor learns length → OOM risk.

**Convergence**: fve_nrm saturates at ~0.64 by step ~150 with winner config (production run).
Remaining 1800 steps give marginal gains — consider cosine LR decay to squeeze more.

## RL infrastructure notes

- `--sglang-disable-radix-cache` — required for input_embeds path
- `del payload["input_ids"]` in `nla_generate.py` — SGLang confused by both ids+embeds. Symptom: k3 (the GRPO importance-ratio diagnostic) floored at ≈0.20; after fix k3≈0.001.
- `--group-rm` for batched critic reward computation (critic_fwd via Ray remote, reward off rollout GPU)
- Step time ~47s with rollout_batch=64×8 (90% is SGLang rollout wait — actor train is 10-20s)
- **`export NLA_EMBED_DUMP_DIR=/dev/shm/nla`** — the per-step 1.1GB embedding dump for
  nla_generate was going to `/tmp` (overlay fs = disk, ~1.5s/step). /dev/shm is tmpfs (RAM).
  Zero code change — `embed_dump_path` already checks this env var.
