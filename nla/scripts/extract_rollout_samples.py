#!/usr/bin/env python
"""Extract formatted rollout samples from an NLA RL dump for qualitative inspection.

Loads a step_NNNN.pt rollout dump (torch.save'd dict with rollout_id + samples list),
picks a mix of best/mid/worst samples by reward, and writes a human-readable report.
"""
import argparse
import textwrap
from pathlib import Path

import numpy as np
import torch


def format_source_box(text: str, width: int = 96) -> str:
    wrapped_lines: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            wrapped_lines.append("")
            continue
        wrapped = textwrap.fill(para, width=width, break_long_words=False, break_on_hyphens=False)
        wrapped_lines.extend(wrapped.split("\n"))
    out = ["  ┌" + "─" * (width + 2)]
    for line in wrapped_lines:
        out.append(f"  │ {line}")
    out.append("  └" + "─" * (width + 2))
    return "\n".join(out)


def extract_explanation_body(response: str) -> str:
    open_tag, close_tag = "<explanation>", "</explanation>"
    if open_tag in response and close_tag in response:
        start = response.index(open_tag) + len(open_tag)
        end = response.index(close_tag)
        return response[start:end].strip()
    return response.strip()


def source_label(doc_id: str) -> str:
    if "WildChat" in doc_id:
        return "WildChat"
    if "Ultra-FineWeb" in doc_id:
        return "UFW"
    return "other"


def pick_indices(rewards: np.ndarray, n_top: int, n_mid: int, n_bot: int) -> list[int]:
    order = np.argsort(-rewards)
    n = len(order)
    top = list(order[:n_top])
    bot = list(order[-n_bot:])
    mid_start = max(0, n // 2 - n_mid // 2)
    mid = list(order[mid_start : mid_start + n_mid])
    seen: set[int] = set()
    picked: list[int] = []
    for idx in top + mid + bot:
        i = int(idx)
        if i not in seen:
            seen.add(i)
            picked.append(i)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_path", help="Path to step_NNNN.pt rollout dump")
    ap.add_argument("--out", required=True, help="Output text file path")
    ap.add_argument("--n-top", type=int, default=6)
    ap.add_argument("--n-mid", type=int, default=8)
    ap.add_argument("--n-bot", type=int, default=6)
    ap.add_argument("--max-src-chars", type=int, default=1800)
    ap.add_argument("--tag", default="", help="Extra tag for header (e.g. 'k2')")
    args = ap.parse_args()

    d = torch.load(args.dump_path, map_location="cpu", weights_only=False)
    step = d["rollout_id"]
    samples = d["samples"]

    rewards = np.array([s["reward"] for s in samples], dtype=np.float64)
    doc_ids = [s["metadata"]["doc_id"] for s in samples]
    n_wild = sum(1 for di in doc_ids if "WildChat" in di)
    n_ufw = sum(1 for di in doc_ids if "Ultra-FineWeb" in di)
    groups = np.array([s["group_index"] for s in samples])
    n_groups = len(np.unique(groups))

    picked = pick_indices(rewards, args.n_top, args.n_mid, args.n_bot)

    hdr = f"NLA RL rollout samples — step {step}"
    if args.tag:
        hdr += f"  [{args.tag}]"
    lines: list[str] = [hdr, "=" * 100]
    lines.append(f"Dump: {args.dump_path}")
    lines.append(f"N samples: {len(samples)}  ({n_groups} GRPO groups × {len(samples) // n_groups})")
    lines.append(f"Split: {n_wild} WildChat  /  {n_ufw} UFW  /  {len(samples) - n_wild - n_ufw} other")
    lines.append(
        f"reward: mean={rewards.mean():.4f}  std={rewards.std():.4f}  "
        f"range=[{rewards.min():.4f}, {rewards.max():.4f}]"
    )
    pct = np.percentile(rewards, [10, 25, 50, 75, 90])
    lines.append(
        f"        p10={pct[0]:.4f}  p25={pct[1]:.4f}  p50={pct[2]:.4f}  "
        f"p75={pct[3]:.4f}  p90={pct[4]:.4f}"
    )
    lines.append("")
    lines.append(
        f"Showing {len(picked)} samples: top {args.n_top} / mid {args.n_mid} / bottom {args.n_bot} "
        f"by reward. Sorted best → worst."
    )
    lines.append("Higher reward = better (lower reconstruction MSE).")
    lines.append("")

    for rank, idx in enumerate(picked):
        s = samples[idx]
        md = s["metadata"]
        reward = float(s["reward"])
        doc_id = md["doc_id"]
        src = md["detokenized_text_truncated"]
        n_tok = md["n_raw_tokens"]
        act_norm = float(np.linalg.norm(np.asarray(md["activation_vector"])))
        lbl = source_label(doc_id)
        expl = extract_explanation_body(s["response"])

        lines.append("━" * 100)
        lines.append(f"[{rank + 1:>2}/{len(picked)}]  reward={reward:.4f}  [{lbl}]  {doc_id}")
        lines.append(
            f"         n_tok={n_tok}  ||act||={act_norm:.2f}  "
            f"resp_toks={s['response_length']}  sample_idx={idx}  group={s['group_index']}"
        )
        lines.append("━" * 100)
        lines.append("")
        lines.append(f"### SOURCE TEXT ###  ({len(src)} chars, {n_tok} tokens — activation @ final token)")
        if len(src) > args.max_src_chars:
            shown = src[: args.max_src_chars]
            trunc_note = f"\n\n[... TRUNCATED: {len(src) - args.max_src_chars} more chars ...]"
            lines.append(format_source_box(shown + trunc_note))
        else:
            lines.append(format_source_box(src))
        lines.append("")
        lines.append("### EXPLANATION (actor output) ###")
        for para in expl.split("\n"):
            if para.strip():
                lines.append(textwrap.fill(para, width=98, initial_indent="  ", subsequent_indent="  "))
            else:
                lines.append("")
        lines.append("")
        lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}  ({len(lines)} lines, {out_path.stat().st_size} bytes)")
    print(
        f"step={step}  N={len(samples)}  reward_mean={rewards.mean():.4f}  "
        f"WildChat={n_wild}  UFW={n_ufw}"
    )


if __name__ == "__main__":
    main()
