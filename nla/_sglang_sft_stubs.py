"""Stub sglang modules so SFT can run without a full sglang+router install.

SFT (--debug-train-only) never touches the sglang engine — it's pure FSDP
training on pre-generated parquets. But miles' top-level imports pull in
sglang, sglang_router, and transitive deps with tight version pins (e.g.
sglang 0.5.x → transformers 4.57+ for GptOssConfig). If your environment has
an older transformers and you can't upgrade, this lets SFT run anyway.

These stubs satisfy the import chain with no-ops. The actual sglang engine
code paths (rollout generation, router) are unreachable in SFT and will
crash loudly if somehow hit. That's the intended failure mode — you'll know
immediately if you accidentally try to use this for RL without a real env.

Usage: in your launcher, before importing train.py:
    import nla._sglang_sft_stubs  # noqa: F401 — sets up sys.modules
"""

import sys
import types


def _stub_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# ─── sglang.srt.constants — string constants, used as offload tags ───
# (real values don't matter for SFT; these just need to be hashable)
_srt = _stub_module("sglang.srt")
_stub_module("sglang").srt = _srt
_constants = _stub_module("sglang.srt.constants")
_constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
_constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
_constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
_srt.constants = _constants

# ─── sglang_router.launch_router.RouterArgs ───
# add_cli_args is called unconditionally in miles.utils.arguments.add_router_arguments
_router = _stub_module("sglang_router")
_launch = _stub_module("sglang_router.launch_router")
_router.launch_router = _launch


class _RouterArgs:
    @staticmethod
    def add_cli_args(parser, use_router_prefix=True, exclude_host_port=True):  # noqa: ARG004
        pass


_launch.RouterArgs = _RouterArgs

# ─── miles.backends.sglang_utils.* — these import sglang engine internals ───
# We replace the whole submodules with stubs. SGLangEngine is wrapped in
# ray.remote() inside _create_rollout_engines — SFT never calls that.
_sglang_utils = _stub_module("miles.backends.sglang_utils")
_engine_mod = _stub_module("miles.backends.sglang_utils.sglang_engine")
_args_mod = _stub_module("miles.backends.sglang_utils.arguments")


class _SGLangEngine:
    def __init__(self, *_, **__):
        raise RuntimeError(
            "SGLangEngine stub — you're trying to run RL rollout but only have "
            "the SFT stubs loaded. Build the miles conda env (build_conda.sh)."
        )


_engine_mod.SGLangEngine = _SGLangEngine
_sglang_utils.sglang_engine = _engine_mod


def _add_sglang_arguments(parser):  # noqa: ARG001
    pass


def _sglang_validate_args(args):  # noqa: ARG001
    pass


_args_mod.add_sglang_arguments = _add_sglang_arguments
_args_mod.validate_args = _sglang_validate_args
_sglang_utils.arguments = _args_mod
