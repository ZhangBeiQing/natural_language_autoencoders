"""Push a local checkpoint directory to remote storage.

Walks the local DCP-format checkpoint dir (model/, optimizer/, hf/ subdirs),
uploads each file via the Storage backend, then writes a _manifest.txt listing
all relative paths. The manifest is written LAST — pull_checkpoint treats its
presence as "upload complete", so a crashed partial upload won't mask a missing
checkpoint.

Fire-and-forget safe: if --local doesn't exist or has no tracker file yet,
prints and exits 0. Intended for nohup'd background use at end-of-training:

    nohup python -m nla.scripts.push_checkpoint \\
        --local /tmp/nla_run/actor \\
        --remote gs://your-bucket/ckpt/actor \\
        --storage-cls my.module.GCSStorage \\
        --only-latest \\
        > /tmp/push_actor.log 2>&1 &
"""

import argparse
import shutil
import sys
from pathlib import Path

from nla.storage import _load_storage

_TRACKER = "latest_checkpointed_iteration.txt"
_MANIFEST = "_manifest.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local", required=True, help="Local checkpoint dir (DCP format)")
    p.add_argument("--remote", required=True, help="Remote destination prefix")
    p.add_argument("--storage-cls", required=True, help="Storage backend class")
    p.add_argument(
        "--only-latest",
        action="store_true",
        help="Push only iter_{latest}/ + tracker, not all intermediate saves",
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help="Treat --local as a flat file dir (no tracker/iter_N structure). "
        "Syncs all files. For rollout_dumps/.",
    )
    args = p.parse_args()

    local_dir = Path(args.local)
    if not local_dir.is_dir():
        print(f"[push_checkpoint] {local_dir} does not exist yet — nothing to push.")
        return

    remote = args.remote.rstrip("/")
    storage = _load_storage(args.storage_cls)

    if args.flat:
        files = sorted(f for f in local_dir.iterdir() if f.is_file())
        if not files:
            print(f"[push_checkpoint --flat] {local_dir} is empty — nothing to push.")
            return
        for f in files:
            dst = f"{remote}/{f.name}"
            storage.ensure_parent(dst)
            with open(f, "rb") as src, storage.open_write(dst) as out:
                shutil.copyfileobj(src, out)
            print(f"  + {f.name} ({f.stat().st_size} bytes)")
        print(f"[push_checkpoint --flat] pushed {len(files)} files → {remote}", file=sys.stderr)
        print(remote)
        return

    tracker = local_dir / _TRACKER
    if not tracker.is_file():
        print(f"[push_checkpoint] no {_TRACKER} in {local_dir} — nothing to push.")
        return

    if args.only_latest:
        latest = int(tracker.read_text().strip())
        iter_dir = local_dir / f"iter_{latest:07d}"
        if not iter_dir.is_dir():
            print(f"[push_checkpoint] tracker says iter={latest} but {iter_dir} missing — nothing to push.")
            return
        iter_dirs = [iter_dir]
        print(f"[push_checkpoint] pushing latest iter={latest} from {local_dir}")
    else:
        iter_dirs = sorted(d for d in local_dir.iterdir() if d.is_dir() and d.name.startswith("iter_"))
        print(f"[push_checkpoint] pushing {len(iter_dirs)} iter dirs from {local_dir}")

    files: list[Path] = []
    for d in iter_dirs:
        files.extend(f for f in d.rglob("*") if f.is_file())
    # Manifest = atomicity requires there IS something to atomically signal.
    # An empty iter_dir (dir created but DCP shards not yet written — race
    # with training's save) would produce a manifest listing only the
    # tracker, which pull restores → training load crashes.
    assert files, (
        f"[push_checkpoint] iter_dirs exist but contain no files — "
        f"likely racing against an in-progress save. "
        f"iter_dirs={[str(d) for d in iter_dirs]}"
    )
    files.append(tracker)

    manifest: list[str] = []
    for f in files:
        rel = f.relative_to(local_dir).as_posix()
        dst = f"{remote}/{rel}"
        storage.ensure_parent(dst)
        with open(f, "rb") as src, storage.open_write(dst) as out:
            shutil.copyfileobj(src, out)
        manifest.append(rel)
        print(f"  + {rel} ({f.stat().st_size} bytes)")

    storage.write_text(f"{remote}/{_MANIFEST}", "\n".join(sorted(manifest)) + "\n")
    print(f"[push_checkpoint] pushed {len(files)} files → {remote}", file=sys.stderr)
    print(remote)


if __name__ == "__main__":
    main()
