"""Metrics-only tis function: observe rollout|train mismatch without changing GRPO.

vanilla_tis_function (loss.py:475) multiplies pg_loss by clamp(ratio) — that's
importance-sampling correction, which changes the objective. This logs the same
metrics + Schulman k3 KL but returns pg_loss unchanged.

Use with:
    --get-mismatch-metrics --custom-tis-function-path nla.tis_metrics.metrics_only
"""

from typing import Any

import torch


def metrics_only(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_lp = torch.cat(rollout_log_probs, dim=0)
    train_lp = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(train_lp - rollout_lp)
    # k3: (r-1) - log(r), nonneg + unbiased (http://joschu.net/blog/kl-approx.html)
    k3 = (tis - 1) - (train_lp - rollout_lp)
    return pg_loss, loss_masks, {
        "tis": tis.detach(),
        "tis_abs": (tis - 1).abs().detach(),
        "tis_k3": k3.detach(),
    }
