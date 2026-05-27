"""Manually fetch a remote parquet + sidecar to local cache.

NLADataSource auto-fetches when prompt_data is remote and --nla-storage-cls
is set, so you usually don't need this. Use it to pre-warm the cache, or to
fetch without a full training launch (e.g. to inspect the parquet locally).

    python -m nla.scripts.fetch_parquet \\
        --remote s3://your-bucket/path/av_sft.parquet \\
        --storage-cls my.module.S3Storage

Prints the local path; that's what training will read from.
"""

import argparse
import sys

from nla.storage import fetch_to_local_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remote", required=True, help="Remote parquet path (s3://, gs://, etc.)")
    p.add_argument("--storage-cls", required=True, help="Storage backend class")
    p.add_argument("--cache-dir", default="/tmp/nla_fetch_cache")
    args = p.parse_args()

    local = fetch_to_local_cache(args.remote, storage_cls=args.storage_cls, cache_dir=args.cache_dir)
    print(f"Fetched to: {local}", file=sys.stderr)
    print(local)  # path only on stdout — script-friendly for $(...)


if __name__ == "__main__":
    main()
