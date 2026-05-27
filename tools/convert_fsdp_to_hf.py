import argparse
import os
import pickle
import shutil
import time

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig, AutoModelForCausalLM
from typing_extensions import override


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _detect_model_dir(input_dir: str) -> str:
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _get_candidate_prefixes(keys: list[str]) -> list[str]:
    predefined = [
        "model_state.model.",
        "model_state.",
        "model.",
        "module.",
        "",
    ]

    detected: set[str] = set()
    for key in keys:
        for prefix in predefined:
            if prefix and key.startswith(prefix):
                detected.add(prefix)

    # Always keep empty string as a fall back option for exact match.
    detected.add("")
    # Preserve predefined order while keeping only detected prefixes.
    return [p for p in predefined if p in detected]


def _strip_best_prefix(keys: list[str], target_keys: set[str]) -> tuple[str, int]:
    best_prefix = ""
    best_match = -1

    for prefix in _get_candidate_prefixes(keys):
        mapped_keys = {k.removeprefix(prefix) for k in keys}
        match_count = len(mapped_keys & target_keys)
        if match_count > best_match:
            best_match = match_count
            best_prefix = prefix

    return best_prefix, best_match


def _resolve_skeleton_config(origin_config, dcp_keys: list[str]):
    """Return the config to build the load-target skeleton from.

    DCP matches by key name, so the skeleton's state_dict keys must match what
    was saved. If origin is a multimodal wrapper (Gemma-3: has .text_config) but
    the DCP was saved from the unwrapped text-only CausalLM (NLATextOnlyCausalLM
    in nla/train_actor.py), the wrapper skeleton's keys are language_model.model.*
    while the DCP has model.* -- zero overlap, load_state_dict(strict=False)
    silently keeps random init. See PORTING_NEW_ARCHITECTURES.md.

    Detection: wrapper config + no DCP key contains the wrapper module prefix
    -> DCP is text-only -> build skeleton from text_config.
    AutoModelForCausalLM._model_mapping[Gemma3TextConfig] -> Gemma3ForCausalLM,
    whose state_dict keys are model.* -- matches the DCP.

    Qwen/Llama/Mistral configs have no .text_config -> pass through unchanged.
    """
    text_config = getattr(origin_config, "text_config", None)
    if text_config is None:
        return origin_config
    if any("language_model." in k for k in dcp_keys):
        return origin_config
    print(
        f"Origin config {type(origin_config).__name__} is a multimodal wrapper but "
        f"DCP has no language_model.* keys -- building text-only skeleton from "
        f"{type(text_config).__name__} (model_type={text_config.model_type})."
    )
    return text_config


def _convert_fsdp_to_hf(
    origin_hf_dir: str,
    input_dir: str,
    output_dir: str,
) -> None:
    print(f"loading FSDP model from {input_dir}")
    t = time.time()
    state_dict = _load_fsdp_state_dict(input_dir)
    print(f"FSDP model loaded in {time.time()-t:.2f} sec.")

    tensor_items = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if not tensor_items:
        raise ValueError(
            "No model weights found in checkpoint. "
            "Please pass the checkpoint directory (e.g. iter_xxx or iter_xxx/model)."
        )

    origin_config = AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=True)
    config = _resolve_skeleton_config(origin_config, list(tensor_items.keys()))
    hf_model = AutoModelForCausalLM.from_config(config)
    target_keys = set(hf_model.state_dict().keys())

    best_prefix, best_match = _strip_best_prefix(list(tensor_items.keys()), target_keys)
    print(
        f"Skeleton: {type(hf_model).__name__} ({len(target_keys)} params). "
        f"Using prefix '{best_prefix}', matched {best_match}/{len(tensor_items)} DCP keys."
    )

    model_state = {k.removeprefix(best_prefix): v for k, v in tensor_items.items()}
    missing, unexpected = hf_model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    assert not missing, (
        f"{len(missing)} skeleton params received no DCP weights -- saved model would "
        f"have random-init garbage. DCP key shape does not match "
        f"{type(hf_model).__name__}. First missing: {missing[:3]}"
    )

    # save_pretrained writes config.torch_dtype from the skeleton (init'd on meta
    # device, never had weights → dtype unset → None). sglang reads None → dtype
    # mismatch: fp32 embeddings hit bf16 weights → RuntimeError at qkv_proj.
    # DCP tensors carry their own dtype; take it from the first one.
    tensor_dtype = next(iter(model_state.values())).dtype
    hf_model.config.torch_dtype = tensor_dtype
    text_cfg = getattr(hf_model.config, "text_config", None)
    if text_cfg is not None:
        text_cfg.torch_dtype = tensor_dtype
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model weights saved to {output_dir} (torch_dtype={tensor_dtype})")


def copy_assets(origin_hf_dir: str, output_dir: str) -> None:
    if not os.path.isdir(origin_hf_dir):
        print(f"Skip copy_assets: {origin_hf_dir} is not a local directory (hub ID?). "
              f"config.json already written by save_pretrained; fetch tokenizer separately if needed.")
        return
    # config.json was written by save_pretrained with the correct (possibly text-only)
    # architectures -- copying origin's multimodal config.json would clobber it.
    skip = {"model.safetensors.index.json", "config.json"}
    for filename in os.listdir(origin_hf_dir):
        if filename in skip or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="The original Hugging Face model directory to load config/tokenizer assets.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    model_dir = _detect_model_dir(args.input_dir)
    _convert_fsdp_to_hf(args.origin_hf_dir, model_dir, args.output_dir)
    copy_assets(args.origin_hf_dir, args.output_dir)
