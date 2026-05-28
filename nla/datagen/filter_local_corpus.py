"""Filter local corpus parquet shards into a cleaner local Stage-0 input parquet."""

import argparse
import glob
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


_DEFAULT_BAD_PATTERNS = [
    r"free shipping",
    r"shipping policy",
    r"add to cart",
    r"\bcart\b",
    r"\bsku\b",
    r"lost your password",
    r"please enter your email address",
    r"please briefly explain why",
    r"report this question",
    r"report this answer",
    r"get access to:",
    r"privacy policy",
    r"cookie policy",
    r"terms of use",
    r"all rights reserved",
    r"email@example\.com",
    r"subscribe to our newsletter",
    r"enable javascript",
]


def _compile_bad_re(extra_patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join([*_DEFAULT_BAD_PATTERNS, *extra_patterns]), re.IGNORECASE)


def _clean_enough(
    text: str,
    score: float | None,
    *,
    score_min: float | None,
    min_chars: int,
    max_chars: int,
    min_words: int,
    max_bullet_line_frac: float,
    bad_re: re.Pattern[str],
) -> bool:
    if score_min is not None and (score is None or score < score_min):
        return False
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if bad_re.search(text):
        return False
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < min_words:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 20:
        bulletish = sum(
            1
            for line in lines
            if line.startswith(("-", "*", "•")) or re.match(r"^\d+[.)]\s", line)
        )
        if bulletish / len(lines) > max_bullet_line_frac:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True, help="input parquet path or glob; repeatable")
    p.add_argument("--output", required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--text-column", default="content")
    p.add_argument("--score-column", default="score")
    p.add_argument("--score-min", type=float, default=0.95)
    p.add_argument("--min-chars", type=int, default=1200)
    p.add_argument("--max-chars", type=int, default=20000)
    p.add_argument("--min-words", type=int, default=220)
    p.add_argument("--max-bullet-line-frac", type=float, default=0.45)
    p.add_argument("--bad-pattern", action="append", default=[], help="extra regex pattern to reject")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--progress-every", type=int, default=5000)
    args = p.parse_args()

    inputs: list[Path] = []
    for item in args.input:
        matches = sorted(glob.glob(item))
        inputs.extend(Path(match) for match in matches)
    if not inputs:
        raise SystemExit("no input parquet files matched")

    bad_re = _compile_bad_re(args.bad_pattern)
    kept_texts: list[str] = []
    kept_scores: list[str | None] = []
    source_files: list[str] = []
    source_rows: list[int] = []
    seen_prefixes: set[str] = set()
    scanned = 0

    for input_path in inputs:
        pf = pq.ParquetFile(input_path)
        columns = [args.text_column]
        has_score = args.score_column in pf.schema_arrow.names
        if has_score:
            columns.append(args.score_column)
        local_row = 0
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=columns):
            texts = batch.column(0).to_pylist()
            scores = batch.column(1).to_pylist() if has_score else [None] * len(texts)
            for text, score_s in zip(texts, scores, strict=True):
                row_idx = local_row
                local_row += 1
                scanned += 1
                if not isinstance(text, str):
                    continue
                try:
                    score = None if score_s is None else float(score_s)
                except (TypeError, ValueError):
                    score = None
                if not _clean_enough(
                    text,
                    score,
                    score_min=args.score_min,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    min_words=args.min_words,
                    max_bullet_line_frac=args.max_bullet_line_frac,
                    bad_re=bad_re,
                ):
                    continue
                prefix = re.sub(r"\s+", " ", text[:500]).lower()
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                kept_texts.append(text)
                kept_scores.append(None if score is None else f"{score:.4f}")
                source_files.append(str(input_path))
                source_rows.append(row_idx)
                if len(kept_texts) == 1 or len(kept_texts) % args.progress_every == 0:
                    print(f"[filter] kept={len(kept_texts)}/{args.length} scanned={scanned}", flush=True)
                if len(kept_texts) >= args.length:
                    break
            if len(kept_texts) >= args.length:
                break
        if len(kept_texts) >= args.length:
            break

    if len(kept_texts) < args.length:
        raise SystemExit(f"only kept {len(kept_texts)} rows after scanning {scanned}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    table = pa.table(
        {
            args.text_column: pa.array(kept_texts, type=pa.string()),
            args.score_column: pa.array(kept_scores, type=pa.string()),
            "source_file": pa.array(source_files, type=pa.string()),
            "source_row": pa.array(source_rows, type=pa.int64()),
        }
    )
    pq.write_table(table, tmp)
    tmp.rename(output)
    print(
        f"[filter] wrote rows={table.num_rows} scanned={scanned} "
        f"size={output.stat().st_size} bytes -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
