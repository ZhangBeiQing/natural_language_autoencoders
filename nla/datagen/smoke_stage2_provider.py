"""Smoke-test the Stage 2 completion provider on real parquet rows."""

import argparse

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
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--provider-cls", default="nla.datagen.providers.MiMoProvider")
    p.add_argument("--provider-kwargs", default=None, help="JSON kwargs for provider constructor")
    args = p.parse_args()

    table = pq.read_table(args.input, columns=["detokenized_text_truncated"])
    rows = min(args.rows, table.num_rows)
    texts = table.column("detokenized_text_truncated").to_pylist()[:rows]
    prompts = [_DEFAULT_INSTRUCTION.format(text=t) for t in texts]

    provider: CompletionProvider = load_class(args.provider_cls)(**parse_kwargs(args.provider_kwargs))
    print(
        f"[smoke-stage2] provider={args.provider_cls} "
        f"model={getattr(provider, 'model', '?')} rows={rows}",
        flush=True,
    )
    raw = provider.complete(prompts)
    assert len(raw) == rows, f"provider returned {len(raw)} completions for {rows} prompts"

    ok = 0
    for i, completion in enumerate(raw):
        cleaned = _extract_and_clean(completion or "", _DEFAULT_RESPONSE_PATTERN)
        feature_count = 0 if cleaned is None else cleaned.count("\n\n") + 1
        raw_len = 0 if completion is None else len(completion)
        print(
            f"[smoke-stage2] row={i} raw_len={raw_len} "
            f"cleaned={cleaned is not None} features={feature_count}",
            flush=True,
        )
        if cleaned is not None and feature_count >= 2:
            ok += 1

    assert ok == rows, f"only {ok}/{rows} completions matched Stage 2 format"
    print(f"[smoke-stage2] OK {ok}/{rows}", flush=True)


if __name__ == "__main__":
    main()
