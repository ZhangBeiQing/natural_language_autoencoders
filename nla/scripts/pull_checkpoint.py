"""Pull a remote checkpoint directory to local filesystem.

Reads the _manifest.txt written by push_checkpoint, downloads each listed
file via the Storage backend. Exits 1 if manifest doesn't exist — shell can
check return code to fall back to retraining:

    if python -m nla.scripts.pull_checkpoint --remote ... --local ... --storage-cls ...; then
        echo "restored from backup"
    else
        echo "no backup, retraining"
    fi
"""

import argparse
import shutil
import sys
from pathlib import Path

from nla.storage import _load_storage

_MANIFEST = "_manifest.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remote", required=True, help="Remote checkpoint prefix (as passed to push)")
    p.add_argument("--local", required=True, help="Local destination dir")
    p.add_argument("--storage-cls", required=True, help="Storage backend class")
    args = p.parse_args()

    remote = args.remote.rstrip("/")
    storage = _load_storage(args.storage_cls)

    manifest_path = f"{remote}/{_MANIFEST}"
    if not storage.exists(manifest_path):
        print(f"[pull_checkpoint] no manifest at {manifest_path} — no backup to restore.", file=sys.stderr)
        sys.exit(1)

    local_dir = Path(args.local)
    rels = [ln for ln in storage.read_text(manifest_path).splitlines() if ln]
    print(f"[pull_checkpoint] pulling {len(rels)} files from {remote}", file=sys.stderr)

    for rel in rels:
        src = f"{remote}/{rel}"
        dst = local_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with storage.open_read(src) as f_src, open(dst, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)
        print(f"  + {rel}", file=sys.stderr)

    print(f"[pull_checkpoint] pulled {len(rels)} files → {local_dir}", file=sys.stderr)
    print(local_dir)


if __name__ == "__main__":
    main()
