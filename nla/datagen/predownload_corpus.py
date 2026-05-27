"""Download a HuggingFace corpus slice to a local parquet before Stage 0.

This keeps network I/O out of the multi-GPU activation extraction path. Run it
once, verify the output parquet exists, then point `corpus.name` at that local
file in the datagen config.
"""

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--corpus-config", default=None)
    p.add_argument("--corpus-split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--retries", type=int, default=20)
    p.add_argument("--retry-sleep", type=float, default=30.0)
    args = p.parse_args()

    split_expr = f"{args.corpus_split}[{args.start}:{args.start + args.length}]"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    last_error: BaseException | None = None
    for attempt in range(1, args.retries + 1):
        print(
            f"[predownload] attempt {attempt}/{args.retries}: "
            f"{args.corpus} config={args.corpus_config or '-'} split={split_expr}",
            flush=True,
        )
        try:
            ds = load_dataset(args.corpus, name=args.corpus_config, split=split_expr)
            assert isinstance(ds, Dataset), f"expected Dataset, got {type(ds).__name__}"
            texts = ds[args.text_column]
            assert len(texts) == args.length, f"expected {args.length} rows, got {len(texts)}"
            tmp = output.with_suffix(output.suffix + ".tmp")
            pq.write_table(pa.table({args.text_column: pa.array(texts, type=pa.string())}), tmp)
            tmp.rename(output)
            print(
                f"[predownload] wrote rows={len(texts)} size={output.stat().st_size} bytes -> {output}",
                flush=True,
            )
            return
        except BaseException as exc:
            last_error = exc
            print(f"[predownload] failed attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            if attempt < args.retries:
                print(f"[predownload] sleeping {args.retry_sleep}s before retry", flush=True)
                time.sleep(args.retry_sleep)

    raise SystemExit(f"failed after {args.retries} attempts: {last_error!r}")


if __name__ == "__main__":
    main()
