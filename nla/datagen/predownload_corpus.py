"""Download a HuggingFace corpus slice to a local parquet before Stage 0.

This keeps network I/O out of the multi-GPU activation extraction path. Run it
once, verify the output parquet exists, then point `corpus.name` at that local
file in the datagen config.
"""

import argparse
import time
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, load_dataset_builder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--corpus-config", default=None)
    p.add_argument("--corpus-split", default=None)
    p.add_argument("--text-column", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--retries", type=int, default=20)
    p.add_argument("--retry-sleep", type=float, default=30.0)
    args = p.parse_args()

    corpus_name = args.corpus.rstrip("/").lower()
    if corpus_name == "openbmb/ultra-fineweb":
        if args.corpus_split is None:
            args.corpus_split = "en"
        if args.text_column is None:
            args.text_column = "content"
    else:
        if args.corpus_split is None:
            args.corpus_split = "train"
        if args.text_column is None:
            args.text_column = "text"

    builder = load_dataset_builder(args.corpus, name=args.corpus_config)
    if builder.info.splits and args.corpus_split not in builder.info.splits:
        available = ", ".join(builder.info.splits)
        raise SystemExit(f"split {args.corpus_split!r} not found; available splits: {available}")
    if builder.info.features and args.text_column not in builder.info.features:
        available = ", ".join(builder.info.features)
        raise SystemExit(f"text column {args.text_column!r} not found; available columns: {available}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    last_error: BaseException | None = None
    for attempt in range(1, args.retries + 1):
        print(
            f"[predownload] attempt {attempt}/{args.retries}: "
            f"{args.corpus} config={args.corpus_config or '-'} split={args.corpus_split} "
            f"start={args.start} length={args.length} text_column={args.text_column}",
            flush=True,
        )
        try:
            ds = load_dataset(args.corpus, name=args.corpus_config, split=args.corpus_split, streaming=True)
            texts = []
            for row_idx, row in enumerate(islice(ds, args.start, args.start + args.length), start=1):
                texts.append(row[args.text_column])
                if row_idx == 1 or row_idx % 100 == 0 or row_idx == args.length:
                    print(f"[predownload] streamed {row_idx}/{args.length} rows", flush=True)
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
