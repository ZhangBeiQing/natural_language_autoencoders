# Setup

## Inference only

You don't need this repo's training stack. Install the runtime deps and use
`nla_inference.py` directly (or the [`kitft/nla-inference`](https://github.com/kitft/nla-inference)
package):

```bash
uv pip install torch transformers safetensors httpx orjson pyyaml numpy
uv pip install "sglang[all]>=0.5.6"
```

> **CUDA/torch pin.** An unpinned `pip install torch` may pull a cu130 build,
> which conflicts with `sgl-kernel`'s cu12 wheels. Either pin `torch<2.11` or
> install with `--index-url https://download.pytorch.org/whl/cu124`.

## Full training stack

NLA is an extension layer on top of Miles + SGLang + (optionally) Megatron-LM.
Clone this repo first — the steps below reference its `nla/miles_patches/` and
`patches/` directories — then install the upstream stack, then this package.

### 0. This repo (clone only)

```bash
git clone https://github.com/kitft/natural_language_autoencoders.git
export NLA_REPO=$PWD/natural_language_autoencoders
```

### 1. Miles

Follow the [Miles installation guide](https://github.com/radixark/miles#installation).
The short version:

```bash
git clone https://github.com/radixark/miles.git
cd miles
git checkout $(cat $NLA_REPO/nla/miles_patches/UPSTREAM_PIN | cut -d@ -f2)
bash build_conda.sh   # creates a conda env with torch, ray, megatron deps
conda activate miles
uv pip install -e .
```

NLA requires the integration patch in `nla/miles_patches/` (adds
`--custom-actor-cls-path`, `--force-use-critic`, and the NLA arg group; see
[docs/design.md §2](design.md)). It is generated against the commit in
`UPSTREAM_PIN` — checking that out first is what makes `git apply` succeed
cleanly:

```bash
cd miles
git apply $NLA_REPO/nla/miles_patches/*.patch
```

> If you skip `build_conda.sh` and `uv pip install -e .` Miles directly into an
> existing env, also install `flash-attn` (`uv pip install flash-attn
> --no-build-isolation`) — Miles' `ring_flash_attn` dependency assumes it.
> Separately, note that installing `sglang[all]` (next step) will pin and
> potentially downgrade `torch` / `transformers` to its tested versions; this
> is expected and the resulting combination is what NLA was developed against.

### 2. SGLang

**Inference only:** stock `uv pip install "sglang[all]>=0.5.6"` is sufficient —
upstream `/generate` already accepts `input_embeds`, and `nla_inference.py`
uses that path unmodified.

**Training:** the rollout loop needs the throughput and correctness fixes in
`patches/` (bf16-base64 transport, chunked-prefill slicing, retract-path KV
fix). These patch source files, so install SGLang from a checkout rather than
a wheel:

```bash
git clone https://github.com/sgl-project/sglang.git
bash $NLA_REPO/patches/apply_sglang_patches.sh ./sglang
uv pip install -e "./sglang/python[all]"
```

(Miles' conda env may already ship an SGLang wheel; the editable install above
shadows it, which is what you want — the patch script expects a repo checkout,
not a `site-packages` layout.)

For Gemma-3 models you must launch the server with `--attention-backend fa3`
(the default flashinfer backend OOMs on `head_dim=256`).

### 3. Megatron-LM (only for the Megatron backend)

The FSDP backend (default for ≤27B) doesn't need this. For the Megatron
backend (used for the 70B Llama run):

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && uv pip install -e .
```

Miles' docs cover the Megatron-specific build flags (transformer-engine, apex).

### 4. This package

```bash
cd $NLA_REPO
uv pip install -e .
```

(Plain `pip install -e .` works everywhere `uv pip` does if you don't have uv.)

The `nla` package will then be importable; the entry points Miles needs are
`nla.train_actor:NLAFSDPActor`, `nla.reward.nla_rm`,
`nla.data_source:NLADataSource`, `nla.rollout.nla_generate.generate`.

### Verify

```bash
python -c "import miles, sglang, nla; print('ok')"
bash configs/critic_sft.sh --help    # should print Miles arg parser
```

## Portability notes

The training pipeline was developed and tested on 8×H100-80GB Linux nodes. It
should run on any reasonably recent CUDA box, but a few environment assumptions
are worth checking:

- **Python env location.** The Miles integration patch sets `PATH`,
  `LD_LIBRARY_PATH`, and `TRITON_PTXAS_PATH` for the SGLang rollout
  subprocesses. These are derived at runtime from `sys.executable` /
  `sys.prefix` / `sysconfig`, so conda, micromamba, and venv all work, as does
  any Python minor version. If your CUDA toolkit is not at `/usr/local/cuda`,
  export `CUDA_HOME` before launching.
- **`/dev/shm` size.** `configs/rl.sh` writes ~1 GB of embedding
  dumps per step to `/dev/shm/nla` (tmpfs, much faster than disk). The Docker
  default of 64 MB is not enough — run containers with `--shm-size=8g`, or
  point `NLA_EMBED_DUMP_DIR` at a disk path.
- **Gated base models.** Gemma-3 and Llama-3.3 require accepting their HF
  license and setting `HF_TOKEN`. Qwen2.5 is ungated and is the recommended
  starting point.
- **GPU layout.** The RL config defaults to 8 actor + 4 critic + 4 rollout GPUs
  (16 total, no colocation). On a single 8-GPU node set
  `ACTOR_GPUS=4 CRITIC_GPUS=2 ROLLOUT_GPUS=2`; Ray will hang on placement if
  the sum exceeds available devices.
