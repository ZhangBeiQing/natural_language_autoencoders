"""Critic-SL rollout: tokenize the AR-SFT prompt string, stash gold activation.

AR-SFT parquet's prompt column is a complete formatted string:
    "Summary of the following text: <text>{explanation}</text> <summary>"

Extraction = last token. The template's fixed suffix ("</text> <summary>")
guarantees the last token is always the extraction point — no PM marker,
no scanning. One-time suffix verification on the first sample catches
template/tokenizer drift early.
"""

import torch

from miles.utils.processing_utils import load_tokenizer

from nla.config import load_nla_config, resolve_sidecar_source, verify_critic_suffix
from nla.schema import MM_ACTIVATION_KEY


_TOKENIZER = None
_SUFFIX_IDS: list[int] | None = None
_SUFFIX_CHECKED = False


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global _TOKENIZER, _SUFFIX_IDS, _SUFFIX_CHECKED
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        cfg = load_nla_config(
            resolve_sidecar_source(
                explicit=getattr(args, "nla_sidecar_source", None),
                hf_checkpoint=args.hf_checkpoint,
                prompt_data=getattr(args, "prompt_data", None),
            ),
            _TOKENIZER,
        )
        _SUFFIX_IDS = cfg.critic_suffix_ids

    samples = data_buffer.get_samples(args.rollout_batch_size)

    for group in samples:
        (sample,) = group
        prompt = sample.prompt
        assert isinstance(prompt, str), (
            f"critic SL requires string prompt (got {type(prompt).__name__}). "
            f"AR-SFT parquet's prompt column should be the complete formatted critic input."
        )

        # add_special_tokens=True matches the stage0 extractor (extractors.py:131)
        # that produced the gold activations. Gemma prepends BOS here; without it
        # the backbone runs OOD and layer-K means diverge from the gold's regime
        # (measurable: init cos(μ_backbone, μ_gold) ~0 vs ~0.9+ with BOS). Qwen has
        # bos_token=None so this is a no-op there — True is correct regardless.
        token_ids = _TOKENIZER(prompt, add_special_tokens=True)["input_ids"]

        # One-time suffix check on first sample — catches template/tokenizer
        # drift before training. Skipped for old sidecars (no suffix_ids).
        if not _SUFFIX_CHECKED and _SUFFIX_IDS is not None:
            verify_critic_suffix(token_ids, _SUFFIX_IDS, context=f"rollout_id={rollout_id}, first sample")
            _SUFFIX_CHECKED = True

        sample.tokens = token_ids
        sample.response_length = 0
        sample.reward = 0.0
        sample.loss_mask = []

        activation = torch.tensor(
            sample.metadata["activation_vector"], dtype=torch.float32
        ).view(1, -1)
        sample.multimodal_train_inputs = {MM_ACTIVATION_KEY: activation}

    return samples
