"""HF-format export for NLA-on-Megatron.

Two jobs, both collective (all ranks must call — PP broadcast + TP all-gather):
  - gather_embedding_for_dump: TP-gather word_embeddings.weight → full [vocab, d]
    for nla_generate's local pre-embed path.
  - save_critic_hf: PP+TP gather all critic params → HF state_dict → safetensors,
    splitting lm_head into value_head.safetensors (our d→d LinearForLastLayer).

Miles already has the complete gather+rename pipeline for the SGLang weight-sync
path. We reuse it: HfWeightIteratorDirect handles PP broadcast + TP all-gather +
Megatron→HF name mapping in one go. We just intercept the NLA-specific params:
  - lm_head.weight: convert_to_hf renames output_layer → lm_head, but ours is
    the d→d value head (LinearForLastLayer), not a real lm_head. Divert to
    value_head.safetensors. (remove_padding inside convert_to_hf tries to slice
    to [:vocab_size], but d < vocab_size for realistic models so it's a no-op.)
  - model.norm.weight: doesn't appear — final_layernorm was replaced with
    Identity() in init(), naturally absent from named_parameters(). HF save
    just omits it. Matches NLACriticModel's stripped-norm arch.
"""

import json
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu
from safetensors.torch import save_file

from miles.backends.megatron_utils.megatron_to_hf.processors.padding_remover import remove_padding
from miles.backends.megatron_utils.update_weight.common import all_gather_param, named_params_and_buffers
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect


_EMBEDDING_WEIGHT = "module.module.embedding.word_embeddings.weight"
# Only weight — bias is dropped in NLAMegatronActor.init() to match FSDP's
# bias=False value_head. convert_qwen2_to_hf doesn't handle output_layer.bias
# anyway (would ValueError).
_LM_HEAD_WEIGHT = "lm_head.weight"


def gather_embedding_for_dump(args, model) -> torch.Tensor | None:
    """TP-gather the embedding weight to full [vocab_hf, d]. Returns on all ranks.

    Collective — every TP rank must call (all_gather_param does a dist.all_gather).
    DP replicates so no DP gather needed. PP: embedding only on pre_process chunks;
    caller gates on is_pipeline_first_stage (this returns None on other stages).
    remove_padding strips Megatron's --make-vocab-size-divisible-by rows so the
    shape matches HF's tokenizer vocab.
    """
    for name, param in named_params_and_buffers(args, model, convert_to_global_name=True):
        if name != _EMBEDDING_WEIGHT:
            continue
        gathered = all_gather_param(args, name, param)
        return remove_padding(name, gathered, args.vocab_size).cpu()
    return None


def save_critic_hf(args, model, hf_config, tokenizer, hf_dir: Path) -> None:
    """Export Megatron NLA critic → HF dir that NLACriticModel.from_pretrained can load.

    Layout matches NLACriticModel.save_pretrained:
      {hf_dir}/model.safetensors          — backbone (truncated, no final norm, no lm_head)
      {hf_dir}/value_head.safetensors     — the d→d LinearForLastLayer
      {hf_dir}/config.json                — with truncated num_hidden_layers
      {hf_dir}/tokenizer*.json            — via tokenizer.save_pretrained

    Collective over PP+TP+EP. HfWeightIteratorDirect's get_hf_weight_chunks does
    the full gather (_get_megatron_full_params: PP broadcast at :70-80, EP at
    :83-98, TP all-gather at :106) then convert_to_hf per param. Same machinery
    as the SGLang weight sync — just redirecting to disk.
    """
    model_name = type(hf_config).__name__.lower()
    iterator = HfWeightIteratorDirect(
        args, model, model_name=model_name, quantization_config=None,
    )
    # The iterator expects a {megatron_name: cpu_tensor} dict (weights_backuper
    # format). We don't have a backuper for the critic (parent early-returns at
    # actor.py:108). Build from live params — _get_megatron_full_params:59 does
    # .to(device=cuda) so passing cuda-resident tensors is a no-op copy.
    live_weights = {
        name: param.detach()
        for name, param in named_params_and_buffers(args, model, convert_to_global_name=True)
    }

    backbone_sd: dict[str, torch.Tensor] = {}
    value_head_sd: dict[str, torch.Tensor] = {}
    for chunk in iterator.get_hf_weight_chunks(live_weights):
        for hf_name, tensor in chunk:
            if hf_name == _LM_HEAD_WEIGHT:
                value_head_sd["weight"] = tensor.to(torch.bfloat16).cpu()
            else:
                backbone_sd[hf_name] = tensor.to(torch.bfloat16).cpu()

    assert "weight" in value_head_sd, (
        f"lm_head.weight not found in gathered params — did NLAMegatronActor "
        f"set args.critic_output_size? convert_to_hf renames output_layer.weight "
        f"→ lm_head.weight; we intercept and divert to value_head. "
        f"Got backbone keys: {sorted(backbone_sd.keys())[:5]}..."
    )
    assert value_head_sd["weight"].shape[0] == value_head_sd["weight"].shape[1], (
        f"expected square d→d value head, got {tuple(value_head_sd['weight'].shape)}. "
        f"remove_padding sliced to vocab_size? That'd mean d > vocab_size — unusual."
    )

    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0 and mpu.is_pipeline_first_stage():
        hf_dir.mkdir(parents=True, exist_ok=True)
        save_file(backbone_sd, str(hf_dir / "model.safetensors"))
        save_file(value_head_sd, str(hf_dir / "value_head.safetensors"))
        # hf_config is ALREADY the truncated K+1-layer config: NLAMegatronActor
        # swaps args.hf_checkpoint = nla_critic_sidecar_source (the FSDP critic
        # HF dir with the right config.json) before parent loads it. Per-layer
        # arrays etc. are already consistent. Assert the swap happened.
        assert hf_config.num_hidden_layers == args.num_layers, (
            f"hf_config.num_hidden_layers={hf_config.num_hidden_layers} != "
            f"args.num_layers={args.num_layers}. NLAMegatronActor.init() should "
            f"have swapped args.hf_checkpoint to the critic's (truncated) HF dir."
        )
        (hf_dir / "config.json").write_text(json.dumps(hf_config.to_dict(), indent=2))
        tokenizer.save_pretrained(hf_dir)
    dist.barrier()
