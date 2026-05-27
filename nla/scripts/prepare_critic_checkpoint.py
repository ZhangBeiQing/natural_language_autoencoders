"""One-time preprocessing: base model → truncated critic-init checkpoint.

Loads a base HF model, keeps blocks 0..K (K+1 layers), strips lm_head + final
LN, saves as a standalone HF checkpoint with config.num_hidden_layers = K+1.
The critic's last_hidden_state is then the output OF block K — exactly what
datagen captured at extraction layer_index K.

After this, Critic-SL's --hf-checkpoint points at the output dir and
NLACriticModel.from_pretrained just works — no layer-count arg needed.

Also writes an nla_meta.yaml sidecar so load_nla_config succeeds. Token IDs
and prompt templates are copied from the DATASET sidecar (the one next to
the parquet you'll train on) since those are dataset-pinned, not model-pinned.

Usage:
    python -m nla.scripts.prepare_critic_checkpoint \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --num-layers 20 \
        --dataset-sidecar path/to/ar_sft.parquet \
        --output /path/to/critic_init
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from nla.config import load_nla_config, write_model_sidecar
from nla.models import NLACriticModel


def _add_megatron_compat_keys(
    output_dir: str, hidden_size: int, vocab_size: int, dtype: torch.dtype
) -> None:
    """mbridge's Qwen2Bridge hard-requires model.norm.weight + lm_head.weight.
    NLACriticModel drops both (Identity norm, d×d value_head separately saved).
    Convert builds output_layer as d×d (critic_output_size=hidden_size), so
    lm_head must be d×d for mbridge scatter to succeed. Eye = identity init,
    matching value_head.safetensors. norm.weight=ones is a no-op under Identity."""
    del vocab_size
    out = Path(output_dir)
    compat_file = "model-megatron-compat.safetensors"
    save_file(
        {
            "model.norm.weight": torch.ones(hidden_size, dtype=dtype),
            "lm_head.weight": torch.eye(hidden_size, dtype=dtype),
        },
        out / compat_file,
    )
    idx_path = out / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    else:
        single = out / "model.safetensors"
        assert single.exists(), f"neither {idx_path.name} nor {single.name} found in {out}"
        with safe_open(single, framework="pt") as f:
            idx = {"metadata": {}, "weight_map": {k: single.name for k in f.keys()}}
    idx["weight_map"]["model.norm.weight"] = compat_file
    idx["weight_map"]["lm_head.weight"] = compat_file
    idx_path.write_text(json.dumps(idx, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True,
                   help="HF checkpoint to truncate (local dir or hub name)")
    p.add_argument("--num-layers", type=int, required=True,
                   help="The datagen extraction layer_index (K). Critic keeps "
                        "blocks 0..K inclusive (num_hidden_layers = K+1 in config.json) "
                        "so last_hidden_state = output of block K = what datagen captured.")
    p.add_argument("--dataset-sidecar", required=True,
                   help="Path to the dataset parquet whose sidecar has token IDs + templates "
                        "(reads {path}.nla_meta.yaml)")
    p.add_argument("--output", required=True, help="Output directory for truncated checkpoint")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--megatron-compat", action="store_true",
                   help="Write dummy model.norm.weight + lm_head.weight so mbridge "
                        "convert_hf_to_torch_dist can handle the non-standard structure. "
                        "NLAMegatronActor replaces both post-load.")
    args = p.parse_args()

    dtype = getattr(torch, args.torch_dtype)

    print(f"Loading {args.base_model} (truncating to {args.num_layers} layers)...")
    model = NLACriticModel.from_pretrained(
        args.base_model,
        nla_num_layers=args.num_layers,
        torch_dtype=dtype,
    )
    # Default nn.Linear init is kaiming_uniform → scales backbone output by ~1/√3,
    # so step-0 pred is near-orthogonal to gold (loss≈2). Identity init means the
    # model starts with pred = backbone_last_hidden (the correct prior — it only
    # has to learn the explanation→activation delta, not undo a random rotation).
    with torch.no_grad():
        model.value_head.weight.copy_(torch.eye(model.config.hidden_size, dtype=dtype))
    print(f"Saving to {args.output}...")
    model.save_pretrained(args.output)
    if args.megatron_compat:
        _add_megatron_compat_keys(
            args.output, model.config.hidden_size, model.config.vocab_size, dtype
        )

    print(f"Loading dataset sidecar from {args.dataset_sidecar}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    cfg = load_nla_config(args.dataset_sidecar, tokenizer)
    assert cfg.d_model == model.config.hidden_size, (
        f"dataset d_model={cfg.d_model} != model hidden_size={model.config.hidden_size}"
    )

    # Bake critic_num_layers into the model sidecar so downstream loads know
    # the truncation without re-reading the dataset sidecar.
    cfg_with_k = replace(cfg, critic_num_layers=args.num_layers)

    print(f"Writing nla_meta.yaml...")
    write_model_sidecar(
        args.output, cfg_with_k,
        role="critic", stage="init",
        base_checkpoint=args.base_model,
        trained_on=[], parent_checkpoints=[args.base_model],
        created_by="nla.scripts.prepare_critic_checkpoint",
    )

    # LlamaTokenizerFast/GemmaTokenizerFast default padding_side='left' (for
    # generation). Downstream reward.py tokenizes with this tokenizer; Megatron's
    # critic_fwd passes attention_mask=None (causal-only) so left-pad tokens would
    # be attended by the last real token. Force right-pad at save so the causal-only
    # assumption holds and the the left-pad fix cumsum fix isn't load-bearing.
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(args.output)
    print(f"Done. Use --hf-checkpoint {args.output} for Critic-SL.")


if __name__ == "__main__":
    main()
