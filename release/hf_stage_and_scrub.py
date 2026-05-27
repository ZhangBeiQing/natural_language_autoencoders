"""Stage an NLA HF checkpoint to a dated dir and scrub for public upload.

Hardlinks weight files (no 100GB+ copy on NFS), copies+scrubs only the small
metadata files. Preserves .cache/huggingface/ resume state across retries.
Release-checks ALL .json/.yaml in the staged dir.

Scrubs:
  nla_meta.yaml — role actor→av/critic→ar, paths→public name, drop rollout_id
  config.json   — drop _name_or_path (and from text_config)

Usage:
    python hf_stage_and_scrub.py SRC_DIR BASE_MODEL CORPUS_DESC LAYER_IDX [PATTERN ...]
    → prints staged dir path on stdout (for shell capture)
"""
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import yaml

INTERNAL_PATH_PATTERNS = ["gs://", "s3://", "/mnt/", "/tmp/", "iter_", "rollout_id"]


def stage_and_scrub(src: str, base_model: str, corpus: str,
                    extra_patterns: list[str],
                    layer_index: int | None = None) -> str:
    src_p = Path(src)
    stage = src_p.parent / f"{src_p.name}_hfstage_{date.today():%Y%m%d}"

    # Hardlink everything from src into stage (cheap, no 100GB NFS copy).
    # If stage exists, refresh links but preserve .cache/ (upload_large_folder
    # resume state). Metadata files are real-copied below before scrubbing
    # so the source is never mutated through the link.
    stage.mkdir(exist_ok=True)
    for f in src_p.iterdir():
        dst = stage / f.name
        if f.is_dir():
            continue  # only top-level files; checkpoints are flat
        if dst.exists():
            dst.unlink()
        os.link(f, dst)

    meta_p = stage / "nla_meta.yaml"
    meta_p.unlink(missing_ok=True)  # break hardlink so write doesn't mutate source
    shutil.copy2(src_p / "nla_meta.yaml", meta_p)
    d = yaml.safe_load(meta_p.read_text())
    d["role"] = {"actor": "av", "critic": "ar"}.get(d.get("role"), d.get("role"))
    d["base_checkpoint"] = base_model
    d["parent_checkpoints"] = [base_model]
    d["trained_on"] = [corpus]
    if "training" in d:
        d["training"].pop("rollout_id", None)
    if layer_index is not None:
        d["extraction_layer_index"] = layer_index
    pt = d.get("prompt_templates", {})
    if "actor" in pt:
        pt["av"] = pt.pop("actor")
    if "critic" in pt:
        pt["ar"] = pt.pop("critic")
    meta_p.write_text(yaml.dump(d, sort_keys=False))

    cfg_p = stage / "config.json"
    cfg_p.unlink(missing_ok=True)
    shutil.copy2(src_p / "config.json", cfg_p)
    cfg = json.loads(cfg_p.read_text())
    cfg.pop("_name_or_path", None)
    if "text_config" in cfg:
        cfg["text_config"].pop("_name_or_path", None)
    cfg_p.write_text(json.dumps(cfg, indent=2))

    # tokenizer*.json are stock upstream vocab files — full of English subwords
    # that false-positive on substring internal-path checks. We never modify them, so
    # exclude from release-check.
    internal_path_patterns = [t.lower() for t in INTERNAL_PATH_PATTERNS + extra_patterns]
    skip = {"tokenizer.json", "tokenizer_config.json"}
    for p in stage.glob("*.json"):
        if p.name not in skip:
            _check(p, internal_path_patterns)
    for p in stage.glob("*.yaml"):
        _check(p, internal_path_patterns)

    return str(stage)


def _check(p: Path, internal_path_patterns: list[str]) -> None:
    txt = p.read_text().lower()
    for pattern in internal_path_patterns:
        assert pattern not in txt, f"internal path in {p}: {pattern!r}"


if __name__ == "__main__":
    src, base_model, corpus, layer_idx, *extra = sys.argv[1:]
    print(stage_and_scrub(src, base_model, corpus, extra,
                          layer_index=int(layer_idx)))
