"""Benchmark Stage-2 provider throughput at different concurrency levels."""

import argparse
import json
import time
from dataclasses import replace

import pyarrow.parquet as pq

from nla.datagen._common import load_class, parse_kwargs
from nla.datagen.providers import CompletionProvider
from nla.datagen.stage2_api_explain import (
    _DEFAULT_INSTRUCTION,
    _DEFAULT_RESPONSE_PATTERN,
    _extract_and_clean,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="raw Stage-1 parquet")
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--concurrency", type=int, action="append", required=True)
    p.add_argument("--provider-cls", default="nla.datagen.providers.MiMoProvider")
    p.add_argument("--provider-kwargs", default="{}", help="JSON kwargs shared by all runs")
    args = p.parse_args()

    base_kwargs = parse_kwargs(args.provider_kwargs)
    table = pq.read_table(args.input, columns=["detokenized_text_truncated"])
    rows = min(args.rows, table.num_rows)
    texts = table.column("detokenized_text_truncated").to_pylist()[:rows]
    prompts = [_DEFAULT_INSTRUCTION.format(text=t) for t in texts]
    provider_cls = load_class(args.provider_cls)

    print(
        json.dumps(
            {
                "provider_cls": args.provider_cls,
                "rows": rows,
                "base_kwargs": {k: v for k, v in base_kwargs.items() if "key" not in k.lower()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for concurrency in args.concurrency:
        kwargs = dict(base_kwargs)
        kwargs["concurrency"] = concurrency
        provider: CompletionProvider = provider_cls(**kwargs)
        started = time.perf_counter()
        try:
            raw = provider.complete(prompts)
            error = None
        except BaseException as exc:
            raw = []
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        cleaned = [_extract_and_clean(x or "", _DEFAULT_RESPONSE_PATTERN) for x in raw]
        ok = sum(x is not None and x.count("\n\n") + 1 >= 2 for x in cleaned)
        none_count = sum(x is None for x in raw)
        print(
            json.dumps(
                {
                    "concurrency": concurrency,
                    "elapsed_s": round(elapsed, 2),
                    "rows": rows,
                    "ok": ok,
                    "none": none_count,
                    "calls_per_s": round((len(raw) or rows) / elapsed, 3) if elapsed > 0 else None,
                    "error": error,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if error:
            break


if __name__ == "__main__":
    main()
