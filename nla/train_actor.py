"""NLAFSDPActor: FSDP 训练 actor，支持按模型类型分发的 NLA 功能。

两个正交维度：
  - 模型类型: LM (.logits) vs NLACriticModel (.values) — 由 _is_critic_model 控制
  - 角色: "actor"（rollout_data 原样使用）vs "critic"（将 actor tokens 替换为 critic tokens）

关键简化：对 critic 模型将 _compute_log_prob 覆写为空操作（因为它们没有 .logits）。
原版 _train_core 即可正常工作 — log_probs/values 为 None 时
compute_advantages_and_returns 提前返回，_train_step 覆写处理 .values。
覆写 _train_core（而非 train），让父类处理 get_rollout_data + 计时器 + 性能日志。
"""

import os
import threading
import shutil
import subprocess
import sys

import ray

from miles.utils.ray_utils import Box
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor, apply_fsdp2
from miles.backends.training_utils.data import get_batch
from miles.backends.training_utils.log_utils import aggregate_forward_results
from miles.backends.training_utils.loss import get_log_probs_and_entropy
from miles.utils.timer import timer
from tqdm import tqdm
from miles.backends.training_utils.loss import loss_function

from nla.arch_adapters import resolve_text_config, resolve_text_model
from nla.config import NLAConfig, load_nla_config_from_args, write_model_sidecar
from nla.injection import inject_at_marked_positions
from nla.models import NLACriticModel, embed_dump_path
from nla.schema import (
    MM_ACTIVATION_KEY, MM_CRITIC_TOKENS_KEY, MM_MSE_SCALE_KEY,
    load_predict_mean_baselines, normalize_activation,
)
from nla.storage import _load_storage, is_remote


CRITIC_ONLY_MM_KEYS = {MM_CRITIC_TOKENS_KEY}


def _swap_rollout_to_critic_tokens(rollout_data: dict, device: torch.device) -> dict:
    """重连 rollout_data: actor tokens → critic tokens，过滤掉解析失败的样本。

    纯数据变换 — 可单元测试。RL rollout 函数将 `nla_critic_tokens`
    （已 tokenize 的 <text>{payload}</text> <summary>{pm}）
    存入 multimodal_train_inputs，仅针对 <explanation> 提取成功的样本。
    缺失 key → 该样本被过滤。

    返回一个新字典。调用者需处理跨 rank 的数量差异 — len(kept) 可能
    每个 rank 不同；参见 _truncate_to_cross_rank_min。
    """
    kept: list[int] = []
    critic_tokens: list[torch.Tensor] = []
    mm_list = rollout_data["multimodal_train_inputs"]
    for i, mm in enumerate(mm_list):
        if mm is None or MM_CRITIC_TOKENS_KEY not in mm:
            continue
        critic_tokens.append(mm[MM_CRITIC_TOKENS_KEY])
        kept.append(i)
    # 此处不对 len(kept) 做 assert — 这是逐 rank 的检查。如果某个 rank 为零
    # 并在此断言失败，其他 rank 会进入 _truncate_to_cross_rank_min 的 all_reduce
    # 并永远挂起。改为让集体断言 n_min > 0 在所有 rank 上同时触发。
    empty_mask = torch.empty(0, dtype=torch.int, device=device)
    return {
        "tokens": critic_tokens,
        "total_lengths": [t.shape[0] for t in critic_tokens],
        "response_lengths": [0] * len(critic_tokens),
        "loss_masks": [empty_mask] * len(critic_tokens),
        "multimodal_train_inputs": [
            {MM_ACTIVATION_KEY: mm_list[i][MM_ACTIVATION_KEY]} for i in kept
        ],
    }


def _assert_reward_train_paths_agree(
    critic_fwd_fn, model: torch.nn.Module, rollout_data: dict, mse_scale: float, tol: float = 0.10
) -> None:
    """Step-0 实时检查: padded critic_fwd MSE == thd-packed 训练 MSE.

    Preflight（rl_preflight.py）在 HF 加载的 critic 上用虚拟数据验证。
    这里在 step 0 用真实的 rollout 数据和真实的后 DCP-overlay critic
    运行一次 — 捕获虚拟批次遗漏的任何问题（例如 DCP 加载损坏的权重，
    或虚拟数据没覆盖到的 tokenizer 边界情况）。

    旧的 left-pad 修复 bug（left-pad + mask.sum-1）会在此产生每样本比率
    ~1.5-2.0 的问题。bf16 GEMM tiling 噪声约 ~1e-4。
    """
    mm_list = rollout_data["multimodal_train_inputs"]
    toks = [mm[MM_CRITIC_TOKENS_KEY] for mm in mm_list if mm and MM_CRITIC_TOKENS_KEY in mm]
    golds = torch.cat([mm[MM_ACTIVATION_KEY] for mm in mm_list if mm and MM_CRITIC_TOKENS_KEY in mm], dim=0)
    # 两条路径在任意子集上必须一致 — 少量不同长度的样本能测试 padding 边界情况；
    # 从 rank 分区中取 32 个样本约增加 ~1s。
    # critic_fwd 返回 .cpu()，但 rollout_data 的 golds 在 rank 的 CUDA
    # 设备上（miles 在数据准备时移动了它们）。统一放在 CPU 上。
    toks, golds = toks[:32], golds[:32].float().cpu()
    n = len(toks)
    if n < 4:
        print(f"[NLA STEP0 CHECK] skipped: n={n} < 4 (smoke-test batch too small for varied-length padding)", flush=True)
        return

    # Reward 路径: pad 到最大长度，attention_mask，critic_fwd 选取 last_idx。
    lens = torch.tensor([t.shape[0] for t in toks])
    T = int(lens.max())
    pad_id = 0  # 永远不会被 attention 关注 — last_idx 在 padding 之前选取
    ids = torch.full((n, T), pad_id, dtype=toks[0].dtype)
    mask = torch.zeros((n, T), dtype=torch.long)
    for i, t in enumerate(toks):
        ids[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = 1
    pred_reward = critic_fwd_fn(ids, mask)  # [n, d] CPU

    # 训练路径: concat，position_ids 在边界处重置，mask=None，
    # use_cache=False 打开 transformers 的 packed-detection 门控。
    packed = torch.cat(toks).unsqueeze(0).cuda()
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lens[:-1].cumsum(0)])
    pos_ids = torch.cat([torch.arange(int(l)) for l in lens]).unsqueeze(0).cuda()
    with torch.no_grad():
        values = model(input_ids=packed, position_ids=pos_ids, attention_mask=None, use_cache=False).values
        pred_train = values[0, (offsets + lens - 1).cuda()].float().cpu()

    def _mse(p: torch.Tensor) -> torch.Tensor:
        pn = normalize_activation(p, mse_scale)
        gn = normalize_activation(golds.float(), mse_scale)
        return ((pn - gn) ** 2).mean(dim=1)

    r = (_mse(pred_reward) / _mse(pred_train)).numpy()
    dev = abs(r - 1.0).max()
    print(f"[NLA STEP0 CHECK] reward/train MSE ratio: mean={r.mean():.4f} max|r-1|={dev:.4f} n={n}", flush=True)
    assert dev < tol, (
        f"step-0 reward-path and training-path MSE diverge by {dev:.1%} (tol {tol:.0%}) on real "
        f"rollout data. Preflight passed — either DCP overlay corrupted the critic, or these "
        f"tokens hit an edge case the dummy prompts missed. Per-sample ratios: {r}"
    )

    # 不检查原始 pred_norm/gold_norm: normalize_activation(v,s) 做的是
    # v/‖v‖·s — MSE loss 是尺度不变的，head 输出范数不受约束。
    # Gemma 的 head 自然以 backbone 尺度输出（~3× gold）。Preflight 的
    # normalize(pred).norm/√d > 0.1 是检查随机方向的正确方式
    #（3月13日 bug）；此 step-0 检查仅覆盖路径分歧。


def _truncate_to_cross_rank_min(
    rollout_data: dict, dp_group, micro_batch_size: int | None
) -> dict:
    """All-reduce len(tokens) 到跨 rank 的最小值，并截断所有列表。

    _swap_rollout_to_critic_tokens 之后，每个 rank 可能有不同的
    len(kept)。get_data_iterator 计算 num_steps = len(tokens) // (gbs/dp)；
    不同长度 → 不同 num_steps → FSDP grad-allreduce 不同步 → 挂起。
    或者使用动态 batching 时，allreduce 中不匹配的 tensor shapes → 挂起。

    同时设置 dynamic_global_batch_size 使 num_steps == 1，不管原始 gbs 是多少。
    """
    n = torch.tensor([len(rollout_data["tokens"])], device=torch.cuda.current_device())
    dist.all_reduce(n, op=dist.ReduceOp.MIN, group=dp_group)
    n_min = n.item()
    if micro_batch_size is not None:
        n_min = (n_min // micro_batch_size) * micro_batch_size
    assert n_min > 0, (
        f"cross-rank min(len(kept)) rounded to {n_min} — at least one rank has "
        f"no valid <explanation> extractions. Actor is not emitting tags "
        f"reliably. Raise rollout_batch_size or check actor SFT checkpoint."
    )
    out = {k: v[:n_min] for k, v in rollout_data.items()}
    out["dynamic_global_batch_size"] = n_min * dist.get_world_size(dp_group)
    return out


def _repartition_for_critic(rollout_data_ref, actor_dp, critic_rank, critic_dp):
    """为具有不同 dp 的 critic 重新分区 actor_dp 切分的 rollout 数据。

    Critic rank i 取 actor 分区 [i, i+critic_dp, i+2*critic_dp, ...]。
    从 Ray 获取这些分区，拼接 list-typed 的 key，重新包装为
    critic_dp 长度的列表，使 process_rollout_data (data.py:273) 看到正确的
    长度。所有 critic rank 共享相同的聚合 total_lengths（它是
    完整数据集的长度），但各自有自己的分区索引。
    """
    assert len(rollout_data_ref) == actor_dp, (
        f"expected {actor_dp} actor partitions, got {len(rollout_data_ref)}"
    )
    my_actor_parts = list(range(critic_rank, actor_dp, critic_dp))
    fetched = [ray.get(rollout_data_ref[i].inner) for i in my_actor_parts]

    # 拼接：分区索引取并集，list-typed 数据 key 追加。
    # total_lengths 是完整数据集的（所有分区相同，保留第一个）。
    merged = {"total_lengths": fetched[0]["total_lengths"], "partition": []}
    for d in fetched:
        merged["partition"].extend(d["partition"])
        for k, v in d.items():
            if k in ("partition", "total_lengths"):
                continue
            if isinstance(v, list):
                merged.setdefault(k, []).extend(v)
            else:
                merged.setdefault(k, v)  # 标量/None: 取第一个

    # 重新包装: critic_dp 个 Box。process_rollout_data 做 refs[dp_rank].inner，
    # 所以只有我们 rank 的 Box 需要真实数据。其他是 None-inner 占位符
    #（永远不会被访问）。Box 类: miles.utils.ray_utils.Box。
    new_refs = [Box(None)] * critic_dp
    new_refs[critic_rank] = Box(ray.put(merged))
    return new_refs


class NLATextOnlyCausalLM:
    """自动类适配 shim: 加载 + 解包多模态模型 → 纯文本 CausalLM。

    miles 的 FSDPTrainRayActor.get_model_cls() 在 hf_config 包含
    vision_config 时返回 AutoModelForImageTextToText（Gemma-3 触发此路径）
    → actor 携带 vision_tower 参数 → RL 权重同步到纯文本 sglang 时
    在第一个 vision_tower key 上返回 400 错误。
    resolve_text_model 解包为仅包含文本侧的 CausalLM wrapper。
    对 Qwen/Llama/Mistral 无影响（无 .language_model 属性）。

    此 shim 接口是 miles 所需的最小接口：`.from_pretrained(...)` 是
    唯一的调用点（fsdp_utils/actor.py: model_cls.from_pretrained(...)）。
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return resolve_text_model(model)


class _SGLangKeyRemap:
    """包装模型，使 state_dict() keys 匹配 sglang 的多模态命名。

    我们的 actor（通过 NLATextOnlyCausalLM）是 Gemma3ForCausalLM — keys 为 `model.*`。
    sglang 将 `google/gemma-3-12b-it` 加载为 Gemma3ForConditionalGeneration
    （HF config 中 architectures=['Gemma3ForConditionalGeneration']） — keys 为
    `language_model.model.*`。权重同步遍历 actor state_dict 并逐字发送
    参数名；sglang 的 load_weights 做 params_dict[name] → KeyError →
    在第一个参数上返回 HTTP 400。

    此 wrapper 仅为权重同步添加前缀（不涉及
    FSDP/训练 — weight_updater 在初始化时捕获自己的模型引用）。
    """

    def __init__(self, model: torch.nn.Module, prefix: str):
        self._model = model
        self._prefix = prefix

    def state_dict(self):
        return {self._prefix + k: v for k, v in self._model.state_dict().items()}


class NLAFSDPActor(FSDPTrainRayActor):

    def init(self, args, role, with_ref=False):
        if role == "critic":
            assert args.critic_save is not None, (
                "NLA RL requires --critic-save (reward fn reads from there)"
            )
            args.hf_checkpoint = args.critic_load
            if args.critic_load_dcp:
                tracker = Path(args.critic_load_dcp) / "latest_checkpointed_iteration.txt"
                assert tracker.is_file(), (
                    f"--critic-load-dcp={args.critic_load_dcp!r} has no tracker file. "
                    f"checkpoint.load() would silently return None → critic keeps "
                    f"HF weights from --critic-load, ignoring the DCP overlay you asked for."
                )
            args.load = args.critic_load_dcp or args.critic_load
            args.save = args.critic_save
            args.lr = args.critic_lr or args.lr
            # Megatron 在 megatron_utils/actor.py:93 处接线此参数；FSDP 不做。
            if args.critic_lr_warmup_iters:
                args.lr_warmup_iters = args.critic_lr_warmup_iters
            args.loss_type = "custom_loss"
            args.custom_loss_function_path = "nla.loss.nla_critic_loss"
            args.nla_model_is_critic = True
            # Critic 的 sidecar 位于 critic_load — 其中包含 critic_num_layers、
            # mse_scale, suffix_ids。命令行上的 --nla-sidecar-source 是 ACTOR 的
            # 覆写（从其模型 sidecar 获取 injection_scale）。切换到 critic 的源：
            # 如果设置了 --nla-critic-sidecar-source 则使用，否则 None →
            # resolve_sidecar_source 回退到 hf_checkpoint = critic_load。
            # （Megatron 必须使用 nla_critic_sidecar_source，因为它的 critic_load
            # 是 torch_dist 格式，没有 sidecar；FSDP 回退到 critic_load
            # HF 目录是 None 可以在这里工作的原因。）
            args.nla_sidecar_source = args.nla_critic_sidecar_source

        self._is_critic_model = getattr(args, "nla_model_is_critic", False)

        rollout_id = super().init(args, role, with_ref)

        # 父类保留完整的 wrapper config（需要 .vision_config 用于其自身
        # 检查）；NLA 只关心文本侧的 hidden_size/num_hidden_layers。
        self._text_config = resolve_text_config(self.hf_config)

        # sglang 加载 HF checkpoint 的架构（Gemma 为多模态），
        # 权重同步发送的是我们文本侧的 keys。如果我们解包了，用
        # 前缀重映射器桥接。weight_updater 仅存在于 actor 角色（sglang
        # 同步路径）— critic 角色不创建它。_text_config != hf_config
        # 是精确的信号：我们已经剥离了多模态 wrapper。
        if (
            not self._is_critic_model
            and self._text_config is not self.hf_config
            and hasattr(self, "weight_updater")
        ):
            arch = (getattr(self.hf_config, "architectures", None) or [""])[0]
            prefix = "language_model." if "ConditionalGeneration" in arch else ""
            if prefix:
                self.weight_updater.model = _SGLangKeyRemap(self.model, prefix)

        assert self.parallel_state.cp_size == 1, (
            "NLA requires cp_size=1. With cp>1, slice_with_cp splits each sample "
            "into non-contiguous chunks; injection token + neighbors can land on "
            "different CP ranks, breaking the in-hook scan."
        )

        # get_grpo_returns (ppo_utils.py) 接收 kl 但仅用于 .ones_like
        #（形状）— 值本身被丢弃。所以 --kl-coef 配合 grpo/gspo 会计算
        # ref_log_probs（很慢！），构建 kl tensor，然后丢弃。实际的
        # GRPO KL 路径是 --use-kl-loss，它将 KL 添加到 policy loss
        # 中（记录为 train/kl_loss）。这在早期的 RL 运行中静默地吞噬了资源。
        if role == "actor" and args.advantage_estimator in ("grpo", "gspo"):
            assert args.kl_coef == 0, (
                f"--kl-coef={args.kl_coef} is a NO-OP under "
                f"--advantage-estimator={args.advantage_estimator}: "
                f"get_grpo_returns discards the kl tensor. Use --use-kl-loss "
                f"--kl-loss-coef {args.kl_coef} instead (adds KL to policy loss, "
                f"logs as train/kl_loss). Or set --kl-coef 0 explicitly if you "
                f"don't want KL."
            )

        if role == "critic" and args.force_use_critic:
            actor_dp = args.actor_num_nodes * args.actor_num_gpus_per_node
            critic_dp = args.critic_num_nodes * args.critic_num_gpus_per_node
            # RolloutManager 按 actor_dp 分区（由 actor rank 0 通过
            # miles/ray/train_actor.py set_train_parallel_config 设置）。Critic 的
            # process_rollout_data 断言 len(refs) == dp_size。
            # 当 dp 不相等时，我们保存 actor_dp 并在 train() 中重新分区。
            assert critic_dp <= actor_dp, (
                f"critic_dp={critic_dp} > actor_dp={actor_dp} is not supported: "
                f"_repartition_for_critic distributes actor partitions across critic "
                f"ranks, so critic ranks >= actor_dp would get nothing. Reduce "
                f"CRITIC_NODES/CRITIC_GPUS so critic_dp <= actor_dp."
            )
            self._nla_actor_dp = actor_dp if actor_dp != critic_dp else None
            if self._nla_actor_dp is not None:
                print(f"[NLA] asymmetric DP: actor={actor_dp} critic={critic_dp}. "
                      f"Critic will fetch all {actor_dp} actor partitions and re-slice.")

        cfg, sidecar_source = load_nla_config_from_args(args, self.tokenizer)
        assert cfg.d_model == self._text_config.hidden_size, (
            f"sidecar d_model={cfg.d_model} != model hidden_size="
            f"{self._text_config.hidden_size}. Wrong checkpoint for this dataset."
        )
        if self._is_critic_model:
            # arguments.py:1796 默认 critic_load=load，所以缺失的
            # --critic-load 会静默加载完整深度的 actor checkpoint。
            # 正面的架构检查会捕获此问题。
            assert cfg.critic_num_layers is not None, (
                f"critic model loaded from {args.hf_checkpoint!r} but sidecar "
                f"has no critic_num_layers. Did --critic-load default to the "
                f"actor checkpoint? Point it at the prepared K+1-layer critic."
            )
            assert self._text_config.num_hidden_layers == cfg.critic_num_layers + 1, (
                f"critic checkpoint has {self._text_config.num_hidden_layers} "
                f"layers, sidecar says extraction layer_index K="
                f"{cfg.critic_num_layers} → expect K+1="
                f"{cfg.critic_num_layers + 1} layers. Wrong checkpoint."
            )

        # injection_scale 是一个训练超参数 — actor 训练必须提供。
        # load_nla_config_from_args 已应用了任何 CLI 覆写；
        # 此处断言该值已解析（通过 CLI、模型 sidecar 或
        # --nla-sidecar-source）。数据集 sidecar 故意不携带
        # injection_scale — 请显式指定。
        #
        # 推理必须匹配：nla_generate.py 也调用 load_nla_config_from_args
        #（相同的辅助函数，相同的解析），因此训练/推理的 scale 不会分叉。
        injects = not self._is_critic_model and args.loss_type in ("sft_loss", "policy_loss")
        if injects:
            assert cfg.injection_scale is not None, (
                "Actor training requires injection_scale. Set --nla-injection-scale "
                "(e.g. '150', 'raw', 'sqrt_d_model'), or point --nla-sidecar-source "
                "at a model sidecar that has it. Dataset sidecars don't carry it — "
                "it's a training hyperparameter, pick explicitly. "
                f"(Resolved sidecar: {sidecar_source!r}, injection_scale: None.)"
            )
        self._nla_cfg: NLAConfig = cfg
        self._nla_vectors: torch.Tensor | None = None
        # 将 mse_scale 暴露在 args 上，使 nla_critic_loss 可以在后端无关的情况下读取。
        # （Megatron 的 forward_step 闭包无法修改 batch；args 是共享通道。）
        self.args.nla_mse_scale = cfg.mse_scale

        # 预测均值的基线值，用于 FVE。如果通过 CLI 传入（--nla-baseline-*，
        # 从 schema.compute_predict_mean_baselines 预计算），则直接使用
        # — 跳过初始化时的 parquet 读取。否则 rank 0 读取 +
        # 广播。Megatron 仅使用 CLI（无回退计算）。
        if self._is_critic_model and args.prompt_data is not None and args.nla_baseline_rawvar is None:
            baselines = [0.0]
            if dist.get_rank() == 0:
                t0 = time.perf_counter()
                source = args.prompt_data.split("@[")[0]
                if is_remote(source):
                    assert args.nla_storage_cls is not None
                    source = _load_storage(args.nla_storage_cls).open_read(source)
                _, b_rv = load_predict_mean_baselines(source, cfg.mse_scale)
                baselines[0] = b_rv
                dt = time.perf_counter() - t0
                print(f"[NLA] FVE baseline rawvar={b_rv:.4f} "
                      f"(mse_scale={cfg.mse_scale}, took {dt:.1f}s)")
            dist.broadcast_object_list(baselines, src=0)
            self.args.nla_baseline_rawvar = baselines[0]

        # miles 调用 gradient_checkpointing_enable() 时不带 kwargs
        # (fsdp_utils/actor.py:123) — HF 默认使用 use_reentrant=True。
        # 可重入式 checkpoint 的 backward 通过自定义
        # autograd.Function 重跑前向，其重计算不触发 FSDP2 的
        # post-forward reshard hook。来自重计算的 all-gather 缓冲区
        # 在 backward 剩余阶段一直保持活跃。62 层 × 826MB
        # (27b) = 51GB 堆积。一旦 adam state 落地，rollout 1 时 74GB OOM。
        # 内存快照 2026-03-13: OOM 时 54 × 826MB foreach_all_gather，
        # 前向只需 17GB (FWDMEM hook) → 仅 backward 时堆积。
        # 不带 grad-ckpt 的独立 FSDP 测试 → 10.64GB → 确认。
        #
        # use_reentrant=False（非可重入）通过普通的
        # 前向调用运行重计算，模块 hook 触发，FSDP 正确 reshard。PyTorch
        # FSDP 文档明确推荐此方式。miles 应在上游修复。
        # NLACriticModel.gradient_checkpointing_enable/_disable 委托
        # 给 backbone，所以两种角色都适用。
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Hook 注册必须在 grad-ckpt 重新启用之后 — 防御性排序。
        # HF 的 gradient_checkpointing_disable() 会清除子模块上的
        # forward hooks（它通过清除 hooks 来移除 checkpoint wrappers），所以
        # 在 disable/re-enable 循环之前注册有丢失 hooks 的风险。
        # 早期的 12b 配置没有 grad-ckpt 重新启用，所以排序无关。
        if not self._is_critic_model and args.loss_type in ("sft_loss", "policy_loss"):
            self._register_injection_hook(self.model)
            if self.ref_model is not None:
                self._register_injection_hook(self.ref_model)

        return rollout_id

    def get_model_cls(self):
        if self._is_critic_model:
            return NLACriticModel
        # NLA 是纯文本的。父类在 hf_config 包含 vision_config 时返回
        # AutoModelForImageTextToText（Gemma-3 触发此路径）→ actor 携带
        # vision_tower 参数 → RL 权重同步到纯文本 sglang 时返回 400 错误。
        # resolve_text_model 解包为 Gemma3ForCausalLM（对 Qwen/Llama 无影响）。
        # 参见 arch_adapters.py。
        return NLATextOnlyCausalLM

    def connect_actor_critic(self, critic_group):
        # Miles 的 PPO critic 创建 actor↔critic NCCL 组来将
        # per-token values 同步到 actor 的 GAE 计算中（megatron_utils/actor.py:552）。
        # NLA 的 critic 是独立的 — GRPO advantages 来自 group-normed
        # rewards，而非 critic values。两组消费相同的 rollout_data_ref；
        # 无需同步。
        pass

    def update_weights(self):
        """将 actor 权重同步到 SGLang，然后 dump embedding 供 nla_generate 使用。

        rollout worker 缓存的 embedding 在每次训练步后会变得过时。
        因为 trainer 在 rollout 期间空闲，且 update_weights 恰在
        rollout 开始前触发，这是 dump 一份新副本的时机。
        nla_generate._maybe_reload_embed 读取它。
        """
        super().update_weights()
        # debug_train_only（SFT 模式）: 无 SGLang rollout worker，所以
        # nla_generate 从不运行 → 没有消费者需要此 dump。跳过 — 节省 ~2.2s/step
        #（FSDP all-gather 1.1GB embedding + torch.save 到磁盘）。
        if (self._is_critic_model or self.args.save is None
                or self.args.debug_rollout_only or self.args.debug_train_only):
            return
        # --offload-train 在此调用前将模型移到 CPU（train.py:92 → sleep()
        # → model.cpu()）。镜像父类 updater 的 .cuda()（update_weight_utils.py:58）
        # 使 .full_tensor() 在 GPU 上运行其 NCCL all-gather。
        weight = self.model.get_input_embeddings().weight.detach().cuda()
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        if dist.get_rank() == 0:
            out_path = embed_dump_path(self.args.save)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            torch.save(weight.detach().cpu(), tmp)
            tmp.rename(out_path)  # 原子操作 — 不会出现写入中途被读取
        dist.barrier()

    def _register_injection_hook(self, model):
        embed = model.get_input_embeddings()
        inj = self._nla_cfg.injection_token_id
        left = self._nla_cfg.injection_left_neighbor_id
        right = self._nla_cfg.injection_right_neighbor_id

        def hook(_module, inputs, output):
            if self._nla_vectors is None or os.environ.get("NLA_SKIP_INJECTION") == "1":
                return output
            assert len(inputs) == 1 and inputs[0].dtype == torch.long
            return inject_at_marked_positions(
                input_ids=inputs[0],
                embeddings=output,
                vectors=self._nla_vectors,
                inj_id=inj, left_id=left, right_id=right,
            )

        embed.register_forward_hook(hook)

    def _get_model_inputs_args(self, batch):
        mm = batch.get("multimodal_train_inputs")
        if mm is not None and MM_ACTIVATION_KEY in mm:
            popped = mm.pop(MM_ACTIVATION_KEY)  # [B, d_model], 从数据集中读取的原始向量
            if self._is_critic_model:
                batch[MM_ACTIVATION_KEY] = popped
                batch[MM_MSE_SCALE_KEY] = self._nla_cfg.mse_scale
            else:
                self._nla_vectors = normalize_activation(popped, self._nla_cfg.injection_scale)
        model_args = super()._get_model_inputs_args(batch)
        # use_cache=False 杀死两个 bug，都通过同一个 DynamicCache：
        #
        # (1) v22 ref_lp: Gemma3TextModel.forward:518 在以下条件时创建 DynamicCache：
        #     `use_cache and past_key_values is None and not self.training`。
        #     ref.eval() → cache; actor (train mode) → none。Gemma3 sliding-window
        #     attn 根据 cache 存在与否选择不同的 mask → ref_lp=-3.39 vs
        #     actor_lp=-1.32，权重完全相同。Qwen 无 sliding window。
        #
        # (2) thd-packed 跨序列污染（miles 在 fsdp_utils/actor.py:645
        #     传递 attention_mask=None）。transformers 有
        #     packed detection — masking_utils.py:735 从 position_id 重置推断
        #     block-diag — 但它被 `past_key_values is None` 作为
        #     门控条件。DynamicCache → 检测被绕过 → SDPA/eager 回退到整个
        #     pack 上的完整 causal mask → seq N 关注 seq 1..N-1。
        #     2026-03-19 验证: eval+default=4.2%
        #     L2 漂移，eval+use_cache=False=0.6%（在 fp32 下 →0.0%，
        #     bf16 GEMM-tiling 来自 batch-shape 差异的噪声）。Qwen FA2:
        #     直接从 position_ids 走 varlen，从不触碰此路径。
        #
        # 请勿移除 — 没有这个，每个 thd microbatch 在 seqs 2..N 上都有
        # 被污染的梯度。所有训练调用者都经过此处
        # (_train_step, _compute_log_prob, _ref_log_probs_no_swap)。
        model_args["use_cache"] = False
        return model_args

    def _create_ref_model(self, ref_load_path):
        # --nla-ref-on-gpu: 自 hook + DynamicCache 修复（两者都在 v12 因为 kl=7.14
        # 症状而被放弃后才提交）以来一直未测试。以 ~7.5GB VRAM 的代价恢复
        # 约 ~20s/step 的 CPU 交换。在生产环境使用前请重新验证 KL init ≈0。
        if not getattr(self.args, "nla_ref_on_gpu", False):
            return super()._create_ref_model(ref_load_path)
        # Miles 硬编码 cpu_offload=True 给 ref → ~20s/step 的 actor↔ref CPU 交换。
        # 在 m16+resp150: actor ~31GB + ref ~7.5GB (FSDP-sharded) = ~38GB，80GB 上完全装得下。
        print(f"[NLA] --nla-ref-on-gpu: ref from {ref_load_path} stays on GPU "
              f"(skips ~20s/step swap, costs ~7.5GB VRAM)")
        with self._get_init_weight_context_manager()():
            ref = self.get_model_cls().from_pretrained(
                ref_load_path, trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
                # convert_fsdp_to_hf 保存为 fp32（DCP 是 fp32）。没有此
                # 类型转换，ref 在 GPU 上占用 2×: 18GB 分片于 DP=6 vs 9GB bf16。
                # 45GB 训练前基线 vs 36 预期 → step 0 时 OOM。
                #（与 CPUOffload 路径的 actor.py:611 相同的修复。）
                torch_dtype=torch.bfloat16,
            )
        full_state = ref.state_dict()
        ref = apply_fsdp2(ref, mesh=self.parallel_state.dp_mesh, cpu_offload=False, args=self.args)
        ref = self._fsdp2_load_full_state_dict(ref, full_state, self.parallel_state.dp_mesh, cpu_offload=False)
        ref.cuda()  # from_pretrained→CPU，FSDP cpu_offload=False 不会移动它 — 现在 pin 到 GPU
        ref.eval()
        return ref

    def _compute_log_prob(self, model_tag, data_iterator, num_microbatches, store_prefix=""):
        # Critic 模型没有 .logits。compute_advantages_and_returns
        # 在 log_probs/values 为 None 时提前返回 (loss.py:315)；get_batch 对
        # 缺失的 key 返回 None (data.py:300)。原版 _train_core 处理其余逻辑。
        # sft_loss (loss.py:785-835) 从 logits 重新计算 — 此遍是浪费的
        #（完整模型前向、注入 hook、clone — ~2× step 时间）。
        if self._is_critic_model or self.args.loss_type == "sft_loss":
            return {}
        if (model_tag == "ref" and self.ref_model is not None
                and getattr(self.args, "nla_ref_on_gpu", False)):
            return self._ref_log_probs_no_swap(data_iterator, num_microbatches, store_prefix)
        return super()._compute_log_prob(model_tag, data_iterator, num_microbatches, store_prefix)

    def _ref_log_probs_no_swap(self, data_iterator, num_microbatches, store_prefix):
        # 与父类 _compute_log_prob 的 ref 分支相同的前向循环，
        # 但移除了 model.cpu()/model.cuda() 交换（ref 已在 GPU 上）。父类版本
        # 在 fsdp_utils/actor.py:310-392 — 这里是那个版本减去第 318-321, 386-392 行。
        forward_data_store = []
        data_iterator.reset()
        with timer(f"{store_prefix}log_probs"), torch.no_grad():
            for step_id in range(len(num_microbatches)):
                for _ in self.prof.iterate_train_log_probs(
                    tqdm(range(num_microbatches[step_id]),
                         desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    batch = get_batch(
                        data_iterator,
                        ["tokens", "loss_masks", "multimodal_train_inputs",
                         "total_lengths", "response_lengths", "max_seq_lens"],
                        self.parallel_state,
                        self.args.data_pad_size_multiplier,
                        self.args.qkv_format,
                        get_position_ids=True,
                    )
                    model_args = self._get_model_inputs_args(batch)
                    logits = self.ref_model(**model_args).logits.float()
                    result = get_log_probs_and_entropy(
                        logits=logits, args=self.args, parallel_state=self.parallel_state,
                        unconcat_tokens=batch["unconcat_tokens"],
                        total_lengths=batch["total_lengths"],
                        response_lengths=batch["response_lengths"],
                        with_entropy=False,
                        max_seq_lens=batch.get("max_seq_lens", None),
                    )
                    forward_data_store.append({f"{store_prefix}log_probs": result["log_probs"]})
        return aggregate_forward_results(forward_data_store, data_iterator, self.args, store_prefix)

    def _train_step(self, batch, step_id, num_microbatches):
        if self._is_critic_model:
            model_args = self._get_model_inputs_args(batch)
            out = self.model(**model_args)
            batch["_nla_backbone_last_hidden"] = out.backbone_last_hidden.detach().squeeze(0)
            values = out.values.float()
            loss, _, log_dict = loss_function(
                self.args, self.parallel_state, batch, num_microbatches, values,
            )
            loss.backward()
            return log_dict
        log_dict = super()._train_step(batch, step_id, num_microbatches)
        # FSDP2 将本 microbatch 的 reduce-scatter 与下一个 microbatch 的
        # all-gather prefetch 重叠执行。Gemma 的 1.41B tied embedding
        # (262k vocab × 5376 d，根 FSDP 组见 fsdp_utils/actor.py:687)
        # 在边界处产生 5 个 embedding 大小的张量同时存活 = 16.9GB 峰值，
        # 叠加在模型 + 优化器之上。Step 0 幸存（无 Adam state）；一旦
        # Adam 存在 (+20GB)，任何 seq-len 方差都会触发此问题。通过
        # torch.cuda.memory._dump_snapshot 诊断 — _fsdp_collectives.py:508
        # foreach_reduce (5.64GB fp32) + :262 foreach_all_gather (2×2.82GB)。
        # Synchronize 在下一次 all-gather 前强制完成 reduce-scatter；
        # 3-tensor 峰值 ~11GB。损失一些通信-计算重叠（~5-10% 吞吐量）。
        # 与 use_reentrant 机制不同（那个在重计算期间破坏 reshard；
        # 这个是 microbatch 之间的边界）。
        torch.cuda.synchronize()
        return log_dict

    def critic_fwd(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """仅推理用的前向，返回每个样本最后一个真实 token 位置的 values。

        在生成期间由 RolloutManager 通过 Ray 调用（此时 trainer 空闲）。
        FSDP 集合操作 — 所有 rank 必须调用此方法（RayTrainGroup.critic_fwd 分发
        到每个 rank）。每个 rank 计算相同输出；调用者取 rank 0 的结果。

        返回 [B, d] CPU tensor（小 → 通过 Ray object store 传回很廉价）。
        """
        assert self._is_critic_model, "critic_fwd called on non-critic actor"
        ids = input_ids.cuda(non_blocking=True)
        mask = attention_mask.cuda(non_blocking=True)
        # mask 中最右边的 True，对任意 padding 方向都鲁棒。
        # GemmaTokenizerFast 默认 padding_side='left'（mask 为
        # [0,0,1,1,1]），旧的 mask.sum-1 会给出 n_real-1 而不是
        # T-1。Qwen 默认 right padding，所以恰好能正常工作。在
        # v19 Gemma RL 中，此问题导致 31/32 样本选错了位置 —
        # actor 追逐了一个人为的长度梯度（更长 → 更少的
        # padding → 更少错误的位置），而非解释质量。
        last_idx = mask.cumsum(dim=1).argmax(dim=1)
        # no_grad，而非 inference_mode：后者将 FSDP 收集的参数
        # tensors 标记为仅推理 → 下一次训练前向在 F.linear 处崩溃
        # 并提示 "Inference tensors cannot be saved for backward"。
        # no_grad 与 autograd 兼容。
        with torch.no_grad():
            values = self.model(input_ids=ids, attention_mask=mask, use_cache=False).values  # [B, T, d]
            out = values[torch.arange(ids.shape[0], device=ids.device), last_idx]  # [B, d]
        return out.float().cpu()

    def train(self, rollout_id, rollout_data_ref):
        # 非对称 DP: RolloutManager 按 actor_dp 分区（由 actor rank 0
        # 通过 miles/ray/train_actor.py set_train_parallel_config 设置）。
        # 当 critic_dp < actor_dp 时，process_rollout_data 断言
        # len(refs) == dp_size → 失败。
        # 在此处重新分区: critic rank i 取 actor 分区 i, i+critic_dp,
        # i+2*critic_dp, ... 。获取+拼接+重新包装。如果 actor_dp %
        # critic_dp != 0（例如 6→4: ranks 0,1 得 2 个分区, 2,3 得 1 个），
        # 负载不均衡，但所有数据都得到处理。当 _nla_actor_dp 为 None 时
        # 无操作（相等 DP，Qwen 的工作路径保持不变）。
        if getattr(self, "_nla_actor_dp", None) is not None:
            rollout_data_ref = _repartition_for_critic(
                rollout_data_ref, self._nla_actor_dp,
                self.parallel_state.dp_rank, self.parallel_state.dp_size,
            )
        return super().train(rollout_id, rollout_data_ref)

    def _train_core(self, rollout_id, rollout_data):
        # 所有数据准备在此处完成 — 父类的 train() 已完成
        # get_rollout_data + 计时器 + 性能日志。
        if self._is_critic_model and self.role == "critic":
            if rollout_id == 0:
                # 所有 rank 都运行（FSDP 前向是集合操作）；每个 rank 看到
                # 自己的 rollout_data 切片，但每样本比率必须全部 ~1.0。
                # 仅在 rank-0 上加断言会导致其他 rank 在 rank 0 死亡时
                # 挂在 FSDP allgather 中 — 让异常在所有地方触发。
                _assert_reward_train_paths_agree(
                    self.critic_fwd, self.model, rollout_data, self._nla_cfg.mse_scale
                )
            rollout_data = _swap_rollout_to_critic_tokens(
                rollout_data, torch.cuda.current_device()
            )
            rollout_data = _truncate_to_cross_rank_min(
                rollout_data,
                self.parallel_state.dp_group,
                None if self.args.use_dynamic_batch_size else self.args.micro_batch_size,
            )
        elif not self._is_critic_model:
            # LM-actor: 剥离变长的 critic tokens（否则会作为
            # multimodal concat 后未知的 kwarg 流入 model(**kwargs)）。
            for mm in rollout_data.get("multimodal_train_inputs") or []:
                if mm is not None:
                    for k in CRITIC_ONLY_MM_KEYS:
                        mm.pop(k, None)
            # _compute_log_prob 截断到 microbatch 边界 (n // micro_bsz * micro_bsz)
            # 但 rollout_data 的每样本列表保持原始长度。对于不可整除的计数
            #（例如 512 rollouts / 3 DP = 170.67 → 171/170/170, 然后
            # 171 // 8 = 21 batches = 168），下游会得到 len(rewards)=171 vs
            # len(log_probs)=168 → IndexError。Qwen 的 batch size 是可整除的。
            #
            # 跨 rank 同步：每个 rank 的 n 可能不同 (171/170/170) → 不同的
            # n_aligned (168/168/168，但此处不保证)。get_data_iterator
            # 读取 dynamic_global_batch_size 来计算 num_microbatches — 如果 ranks
            # 不一致，某个 rank 会得到 []（all_reduce MIN 挂起或断言失败）。
            # 与上方的 critic _truncate_to_cross_rank_min 相同的模式。
            n_local = torch.tensor(
                [len(rollout_data.get("tokens", []))],
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(n_local, op=dist.ReduceOp.MIN, group=self.parallel_state.dp_group)
            micro = self.args.micro_batch_size
            n_aligned = (n_local.item() // micro) * micro
            assert n_aligned > 0, (
                f"actor has {n_local.item()} samples after cross-rank MIN, "
                f"fewer than micro_batch_size={micro}. Raise rollout_batch_size."
            )
            n_orig = len(rollout_data.get("tokens", []))
            for k, v in list(rollout_data.items()):
                if isinstance(v, list) and len(v) == n_orig:
                    rollout_data[k] = v[:n_aligned]
            rollout_data["dynamic_global_batch_size"] = (
                n_aligned * dist.get_world_size(self.parallel_state.dp_group)
            )
        super()._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
        self._nla_vectors = None

    def save_model(self, rollout_id, force_sync=False):
        super().save_model(rollout_id, force_sync)
        if self.args.debug_rollout_only or self.args.save is None:
            return

        # get_model_state_dict 配合 full_state_dict=True 是一个集合操作。
        # 所有 rank 必须调用它，否则 rank 0 会在 all-gather 中死锁。
        # actor.py:96 不传 torch_dtype → 模型保存为 fp32；
        # MixedPrecision 仅影响计算。此处转换为 bf16 以节省 2× 空间。
        full_sd = None
        if self._is_critic_model:
            full_sd = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            full_sd = {
                k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v)
                for k, v in full_sd.items()
            }

        # 匹配 fsdp_utils/checkpoint.py:199 的 iter_{rollout_id+1} 约定。
        iter_dir = f"{self.args.save}/iter_{rollout_id + 1:07d}"

        if dist.get_rank() == 0:
            if self._is_critic_model:
                hf_dir = f"{iter_dir}/hf"
                self.model.save_pretrained(hf_dir, state_dict=full_sd)
                self.tokenizer.save_pretrained(hf_dir)
                # 在 hf/ 和 iter_N/ 两处都写入 sidecar — hf/ 是
                # --critic-load 需要的（与 config.json 并列），iter_N/
                # 是防误用的（如果有人指向了错误的层级）。
                self._write_sidecar(hf_dir, rollout_id)
                self._write_sidecar(iter_dir, rollout_id)
            else:
                # Actor 仅保存 DCP（model/, optimizer/）。加载 RL 时：
                #   --hf-checkpoint = 基础模型（from_pretrained 需要 safetensors）
                #   --load = 此 iter_dir（DCP 覆盖权重）
                #   --nla-sidecar-source = 此 iter_dir（从 sidecar 获取 injection_scale）
                # SGLang 也加载基础模型；update_weights 通过 NCCL 同步 SFT 权重。
                self._write_sidecar(iter_dir, rollout_id)
            keep_n = max(2, int(os.environ.get("NLA_KEEP_LOCAL", "2")))
            save_dir = self.args.save

            def _bg():
                self._maybe_background_push()
                if os.environ.get("NLA_BACKUP_REMOTE"):
                    prune = (f"ls -1d {save_dir}/iter_* 2>/dev/null | "
                             f"head -n -{keep_n} | xargs -r rm -rf")
                    subprocess.run(["bash", "-c", prune], check=False)

            threading.Thread(target=_bg, daemon=True).start()
        dist.barrier()

    def _maybe_background_push(self):
        """每次 checkpoint 后，如果设置了 NLA_BACKUP_REMOTE，则 fire-and-forget 推送到 GCS。

        对于 gs:// 远程，直接使用 gsutil -m cp -r（无需额外依赖）。
        对于其他 scheme，通过 push_checkpoint + storage_cls 处理。
        start_new_session 分离进程 — 上传在训练器后续被 pkill 后仍可存活。
        """
        remote = os.environ.get("NLA_BACKUP_REMOTE")
        if not remote:
            return
        # 与上一次 push 串行化：save_model 的 _bg 线程在此返回后立即
        # 修剪旧的 iter_*。以 keep_n=2 和落后一步的推送，
        # save N+1 的修剪删除 iter_{N-1} — 而 save N 的 gsutil 可能仍在
        # 上传 → 截断/损坏远程文件。阻塞直到该上传
        # 完成；下一次修剪才是安全的。
        prev = getattr(self, "_push_proc", None)
        if prev is not None:
            prev.wait()
        role = "critic" if self._is_critic_model else "actor"
        remote_dir = f"{remote}/{role}"
        log = f"/tmp/push_{role}_iter.log"
        tracker = Path(self.args.save) / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            # 首次保存配合 --async-save: super() 启动了写入，之前没有记录
            # 需要完成 → 还没有 tracker。推送按设计落后一步；首次保存
            # 没有内容可推送。
            return
        latest = tracker.read_text().strip()
        iter_dir = f"{self.args.save}/iter_{int(latest):07d}"
        if remote.startswith("gs://"):
            if role == "actor":
                train_log = os.environ.get("NLA_TRAIN_LOG")
                if train_log and Path(train_log).exists():
                    shutil.copy(train_log, f"{iter_dir}/train.log")
                # RolloutManager 将 sample_offset/epoch_id 写入
                # SIBLING rollout/ 目录 — 不在 iter_dir 内，所以 gsutil
                # cp -r iter_dir 会漏掉它。将其快照到 iter_dir 中，从而从 GCS
                # 恢复时能恢复数据偏移量（否则新 pod → 在前 ~128k 行上反复迭代）。
                rollout_state_dir = Path(self.args.save) / "rollout"
                if rollout_state_dir.exists():
                    for f in rollout_state_dir.glob("global_dataset_state_dict_*.pt"):
                        shutil.copy(f, f"{iter_dir}/{f.name}")
            # env -u PYTHONPATH: 训练的 PYTHONPATH（Megatron-LM checkout）
            # 泄漏到 nix gsutil 的子进程中 → boto 的
            # platform.python_version() 在 conda-forge 的 sys.version 字符串上出错。
            # 仅推送 — 调用者处理修剪（两个后端：save_model 中的
            # daemon 线程，push-then-prune）。之前在 async-save
            # 后台写饱和时，此处链式调用 prune 导致 `ls` readdir 挂起。
            cmd = ["bash", "-c",
                   f"env -u PYTHONPATH gsutil -m cp -r {iter_dir} {remote_dir}/"]
        else:
            storage_cls = os.environ.get("NLA_BACKUP_STORAGE_CLS")
            assert storage_cls, "NLA_BACKUP_STORAGE_CLS required for non-gs:// remote"
            cmd = [sys.executable, "-m", "nla.scripts.push_checkpoint",
                   "--local", self.args.save, "--remote", remote_dir,
                   "--storage-cls", storage_cls, "--only-latest"]
        self._push_proc = subprocess.Popen(
            cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"[NLA] background push fired: {iter_dir} → {remote_dir} (log: {log})")

    def _write_sidecar(self, checkpoint_dir: str, rollout_id: int):
        cfg = self._nla_cfg
        if self._is_critic_model:
            # num_hidden_layers 是 K+1（包含 blocks 0..K — 我们需要
            # block K 的输出）。Sidecar 存储 K（提取层 layer_index，
            # 与 datagen 的 extraction.layer_index 约定一致）。
            cfg = replace(cfg, critic_num_layers=self._text_config.num_hidden_layers - 1)
        write_model_sidecar(
            checkpoint_dir, cfg,
            role="critic" if self._is_critic_model else "actor",
            stage="rl" if self.args.loss_type == "policy_loss" else "sl",
            base_checkpoint=self.args.hf_checkpoint,
            trained_on=[self.args.prompt_data] if self.args.prompt_data else [],
            parent_checkpoints=[self.args.hf_checkpoint],
            created_by="nla.train_actor.NLAFSDPActor",
            training_args={
                "rollout_id": rollout_id,
                "lr": self.args.lr,
                "loss_type": self.args.loss_type,
                "global_batch_size": self.args.global_batch_size,
            },
        )
