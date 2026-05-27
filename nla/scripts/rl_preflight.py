"""RL preflight: catch train/reward path divergence before ray spins up.

Three checks, each targeting a real past bug:

  1. reward-path MSE ≡ training-path MSE (the left-pad fix, Mar 19)
       GemmaTokenizerFast defaults padding_side='left' → critic_fwd's
       mask.sum-1 picked the wrong position for 31/32 samples. Reward
       was Spearman=0.57 noise; actor spent 200 steps chasing an
       artificial length gradient. Qwen worked by accident (right-pad).

  2. critic predictions at plausible scale (Mar 13 random-weights bug)
       resolve_text_model returned the wrong wrapper → load key mismatch
       → critic head stayed at randn init. pred_norm would be ~100× off
       the mse_scale-implied scale. Cost: days of RL on noise rewards.

  3. actor injection hook reaches the forward-path embed ( era)
       FSDP wrapping / multimodal unwrapping can leave get_input_embeddings()
       pointing at an orphan module. Hook registers, never fires, actor
       silently trains without injection. Only run if --actor-hf-dir given.

Single GPU, ~30s total. Run after assets exist, before ray.

Backend coverage:
  Checks 1-2 are backend-agnostic — both FSDP and Megatron critics export
  to HF format (Megatron via save_critic_hf), and NLACriticModel loads either.
  A Megatron critic_output_size wiring bug surfaces here as a missing or
  wrong-shape value_head.safetensors → from_pretrained or pred-shape failure.

  Check 3 is FSDP-only (HF's get_input_embeddings() API). Megatron's
  LanguageModelEmbedding hook + TP/SP seq_slice + output_layer swap need a
  live GPTModel with distributed init — too heavy for preflight. Those are
  verified at NLAMegatronActor.init() time instead: structural asserts on
  gpt.get_submodule("embedding") and the output_layer surgery fail loudly
  on first rank before any training step.
"""
import argparse
import json
import math
import sys

import torch
import yaml
from safetensors import safe_open
from transformers import AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import normalize_activation
from nla.train_actor import NLATextOnlyCausalLM


def _die(msg: str) -> None:
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_critic_paths_and_scale(critic_hf_dir: str, tol: float) -> None:
    tok = AutoTokenizer.from_pretrained(critic_hf_dir)
    print(f"tokenizer: {type(tok).__name__}, padding_side={tok.padding_side!r}")
    cfg = load_nla_config(critic_hf_dir, tok)
    d_model, mse_scale = cfg.d_model, cfg.mse_scale

    # v21's critic was prepared with --num-layers 33 when extraction
    # layer_index=32 → num_hidden_layers=34 (one too many). Head had to
    # approximately undo block 33's transform to hit the gold at block-32
    # output. SFT-FVE ceiling dropped to ~0.32. --num-layers must equal the
    # extraction layer_index K (script keeps 0..K inclusive = K+1 layers).
    hf_cfg = json.load(open(f"{critic_hf_dir}/config.json"))
    n_layers = hf_cfg.get("num_hidden_layers") or hf_cfg.get("text_config", {}).get("num_hidden_layers")
    meta = yaml.safe_load(open(f"{critic_hf_dir}/nla_meta.yaml"))
    k = (meta.get("critic") or {}).get("extraction_layer_index")
    print(f"critic num_hidden_layers={n_layers}, sidecar extraction_layer_index={k}")
    assert k is not None, (
        f"critic sidecar at {critic_hf_dir!r} missing critic.extraction_layer_index. "
        f"schema v2 requires this field — can't verify layer truncation without it. "
        f"Check prepare_critic_checkpoint wrote the sidecar correctly."
    )
    assert n_layers == k + 1, (
        f"critic truncated to {n_layers} layers but extraction layer_index={k} "
        f"→ want {k+1}. Off-by-one in prepare_critic_checkpoint --num-layers."
    )

    # Varied lengths so padding is non-trivial. Short-to-long spread
    # maximizes left-pad offset error in the old bug.
    dummies = [
        "explain: cat",
        "explain: the quick brown fox jumps over the lazy dog repeatedly",
        "explain: a b c d e f g h i j k l m n o p q r s t u v w x y z " * 3,
        "explain: singular",
    ]
    n = len(dummies)
    gold = torch.randn(n, d_model, dtype=torch.float32)

    m = NLACriticModel.from_pretrained(critic_hf_dir, torch_dtype=torch.bfloat16)
    m.cuda().eval()

    # ─── reward path: padded batch, critic_fwd index math ───
    enc = tok(dummies, add_special_tokens=True, padding=True, return_tensors="pt")
    ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
    last_idx = mask.cumsum(dim=1).argmax(dim=1)  # rightmost True, the left-pad fix
    with torch.no_grad():
        values = m(input_ids=ids, attention_mask=mask, use_cache=False).values
        pred_rwd = values[torch.arange(n, device=ids.device), last_idx].float().cpu()

    # ─── training path: thd packing (concat, position_ids with resets) ───
    per = [tok(d, add_special_tokens=True, return_tensors="pt")["input_ids"][0] for d in dummies]
    lens = [int(t.shape[0]) for t in per]
    offsets = torch.tensor([0] + lens[:-1]).cumsum(0)
    packed = torch.cat(per).unsqueeze(0).cuda()
    position_ids = torch.cat([torch.arange(l) for l in lens]).unsqueeze(0).cuda()
    with torch.no_grad():
        values = m(input_ids=packed, position_ids=position_ids, attention_mask=None, use_cache=False).values
        picks = (offsets + torch.tensor(lens) - 1).cuda()
        pred_train = values[0, picks].float().cpu()

    # normalize_activation(x, s) brings x to unit-ish per-dim scale by
    # design (s chosen so real golds land near ‖·‖≈√d). A trained head
    # predicts in that regime; a random head outputs near-zero. Gemma's
    # Mar 13 random-init bug would give pred_per_dim ~0.01 here.
    pred_per_dim = pred_rwd.norm(dim=1).mean() / math.sqrt(d_model)
    print(f"critic pred norm (normalized, per-dim): {pred_per_dim:.3f}")
    if pred_per_dim < 0.1:
        _die(
            f"critic predictions near zero (per-dim {pred_per_dim:.3f}) — head likely at "
            f"random init. Check NLACriticModel.from_pretrained key matching."
        )

    def mse(pred: torch.Tensor) -> torch.Tensor:
        pn = normalize_activation(pred, mse_scale)
        gn = normalize_activation(gold, mse_scale)
        return ((pn - gn) ** 2).mean(dim=1)

    ratio = (mse(pred_rwd) / mse(pred_train)).numpy()
    max_dev = abs(ratio - 1.0).max()
    print(f"per-sample MSE ratio (reward/train): {ratio}")
    print(f"max |ratio - 1| = {max_dev:.4f}")
    if max_dev > tol:
        # The left-pad bug gave ratio ~1.5-2.0 on varied-len batches.
        _die(
            f"reward and training MSE diverge by {max_dev:.1%} on a frozen critic. "
            f"Check last_idx vs padding_side, use_cache gate (DynamicCache bypasses "
            f"packed-detection), or normalize_activation/mse_scale drift."
        )
    print(f"PASS: reward-path ≡ training-path MSE within {tol}")


def check_megatron_critic_export(critic_hf_dir: str, d_model: int) -> None:
    """Verify save_critic_hf wrote a square d→d value head.

    Catches critic_output_size wiring: if model_provider didn't swap
    output_layer (role!="critic" and nla_model_is_critic not set), the
    head stays [vocab, d] → export either writes wrong shape or skips
    value_head entirely. Explicit check here fails faster and clearer
    than the from_pretrained load inside check_critic_paths_and_scale.
    """
    with safe_open(f"{critic_hf_dir}/value_head.safetensors", framework="pt") as f:
        shape = tuple(f.get_slice("weight").get_shape())
    print(f"value_head.safetensors weight shape: {shape}")
    assert shape == (d_model, d_model), (
        f"expected square [{d_model}, {d_model}] value head, got {shape}. "
        f"model_provider didn't swap output_layer → check args.nla_model_is_critic "
        f"was set before super().init()."
    )
    print("PASS: Megatron critic export has correct value_head shape")


def check_actor_hook_fires(actor_hf_dir: str) -> None:
    # CPU, meta-ish — single token, hook fire count is all we need.
    # Mirrors train_actor.py's production load+register path so any
    # unwrapping/orphan regression breaks here first.
    m = NLATextOnlyCausalLM.from_pretrained(actor_hf_dir, torch_dtype=torch.bfloat16)
    m.eval()
    embed = m.get_input_embeddings()
    fires = [0]

    def hook(_mod, _inp, out):
        fires[0] += 1
        return out

    embed.register_forward_hook(hook)
    with torch.no_grad():
        m(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    print(f"actor hook fires: {fires[0]} (embed={type(embed).__name__})")
    if fires[0] == 0:
        _die(
            f"get_input_embeddings() returned a module the forward pass never "
            f"calls. Check resolve_text_model unwrapping for {type(m).__name__}."
        )
    print("PASS: actor injection hook reaches the forward-path embed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--critic-hf-dir", required=True)
    p.add_argument("--actor-hf-dir", default=None)
    p.add_argument("--backend", choices=["fsdp", "megatron"], default="fsdp")
    p.add_argument("--tol", type=float, default=1e-3)
    args = p.parse_args()

    if args.backend == "megatron":
        tok = AutoTokenizer.from_pretrained(args.critic_hf_dir)
        cfg = load_nla_config(args.critic_hf_dir, tok)
        check_megatron_critic_export(args.critic_hf_dir, cfg.d_model)

    check_critic_paths_and_scale(args.critic_hf_dir, args.tol)

    if args.actor_hf_dir:
        assert args.backend == "fsdp", (
            "--actor-hf-dir tests HF's get_input_embeddings() hook path (FSDP only). "
            "Megatron's LanguageModelEmbedding hook is structurally verified at "
            "NLAMegatronActor.init() — gpt.get_submodule('embedding') raises on "
            "first rank if the module path is wrong."
        )
        check_actor_hook_fires(args.actor_hf_dir)


if __name__ == "__main__":
    main()