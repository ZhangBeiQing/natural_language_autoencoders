"""Stage 2: API explanations — SL subset → +api_explanation column.

Call an external LLM to produce natural-language explanations of each row's
source text (`detokenized_text_truncated`). The explanation becomes the
`response` for AV-SFT (actor SFT) and the `<text>` content for AR-SFT (critic SFT).
RL subset skips this stage — actor generates responses during rollout.

The completion backend is pluggable via CompletionProvider. Default: Anthropic.

Processes in chunks for bounded memory + progress visibility. Each completed
chunk is written to {output}.chunks/chunk_{N}.parquet immediately — restart
skips existing chunk files, so a crash at chunk 150/489 loses only that
chunk's API calls. At the end, chunks are concatenated into the output.
"""

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs
from nla.datagen.prompt_cache import load_explanation_cache, lookup
from nla.datagen.providers import CompletionProvider
from nla.datagen.sidecar import NLAApiSummaryMeta, read_sidecar, write_sidecar

# Instruction prompt adapted from the prompt template in the appendix — the proven
# NLA prompt. Shortened from 4-5 → 2-3 features and ~100 word budget so
# responses reliably fit in the token budget WITH closing tag (truncated
# responses fail the extract pattern and get dropped).
_DEFAULT_INSTRUCTION = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~10-20 word description. Feel free to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"

The final feature must describe the very end of the presented sequence: its role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~80-100 words total, put a blank line between each feature, and ALWAYS close the tag:
<analysis>
[first feature — include specific examples when relevant]

[second feature]

[final feature: the last token, its role, immediate constraints]
</analysis>

Text to analyze:

<begin_text>{text}<end_text>"""

# Strict: both opening and closing tags MUST be present. Truncated responses
# (max_tokens cut off before </analysis>) fail this and get dropped — we'd
# rather lose the row than train on half a thought.
_DEFAULT_RESPONSE_PATTERN = r"<analysis>\s*(.*?)\s*</analysis>"

# Minimum features required — the prompt asks for 2-3, so fewer than 2 means
# the model ignored format. These rows are dropped.
_MIN_FEATURES = 2

# Prefix stripping — API models use all kinds of list markers. We want plain
# paragraphs separated by \n\n, no formatting.
_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[-*•+–—]"              # bullet chars (incl. en/em dash)
    r"|\d+[.)]"              # 1. 1)
    r"|\(\d+\)"              # (1)
    r"|[a-zA-Z][.)]"         # a. a) A. A)
    r"|\([a-zA-Z]\)"         # (a) (A)
    r"|[ivxIVX]+[.)]"        # i. ii) IV.
    r")\s+"
)
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")
_INLINE_FEATURE_HEADER_RE = re.compile(
    r"\s+(?=("
    r"Second"
    r"|Final feature"
    r"|Final token"
    r"|Final word"
    r"|The final (?:token|word)"
    r"|Immediate syntactic constraint"
    r"|Semantic topic continuation"
    r"|Semantic expectation"
    r"|Syntactic expectation"
    r"|Syntactic/structural (?:constraint|constraints)"
    r"|Immediate semantic expectations?"
    r"|Stylistic/register(?: patterns?)?"
    r"|Narrative/argumentative momentum"
    r"|Domain/genre signals?"
    r"|Repetition/continuation patterns?"
    r")\s*:)",
    flags=re.IGNORECASE,
)
_INLINE_ORDINAL_RE = re.compile(r"\s+(?=(?:Second|Third|Finally),)", flags=re.IGNORECASE)
_INLINE_FINAL_PHRASE_RE = re.compile(
    r"\s+(?=(?:Semantic expectation|The final token|The final word)\b)",
    flags=re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"`])")


def _extract_and_clean(raw: str, pattern: str) -> str | None:
    """Extract content inside response tags, strip list formatting, emit \\n\\n-separated paragraphs.

    Returns None if the pattern doesn't match (truncated, no tags) — caller drops the row.
    """
    m = re.search(pattern, raw, flags=re.DOTALL)
    if m is None:
        return None
    content = _INLINE_FEATURE_HEADER_RE.sub("\n", m.group(1))
    content = _INLINE_ORDINAL_RE.sub("\n", content)
    content = _INLINE_FINAL_PHRASE_RE.sub("\n", content)

    cleaned: list[str] = []
    for line in content.split("\n"):
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_WRAP_RE.sub(r"\1 ", line)  # **Header:** text → Header: text
        line = line.strip().strip("*_")  # trailing emphasis markers
        if line:
            cleaned.append(line)
    if len(cleaned) == 1:
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(cleaned[0]) if s.strip()]
        if len(sentences) >= _MIN_FEATURES:
            cleaned = sentences
    return "\n\n".join(cleaned)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="SL subset parquet from stage1")
    p.add_argument("--output", required=True)
    p.add_argument("--provider-cls", default="nla.datagen.providers.AnthropicProvider")
    p.add_argument("--provider-kwargs", default=None, help="JSON dict of extra kwargs for the provider constructor")
    p.add_argument("--instruction-template", default=_DEFAULT_INSTRUCTION,
                   help="prompt template with {text} placeholder")
    p.add_argument("--response-extract-pattern", default=_DEFAULT_RESPONSE_PATTERN,
                   help="regex with one capture group — extracts content from API response. "
                        "Default requires both <analysis> and </analysis> (truncated "
                        "responses are dropped). MUST match the tag your instruction asks for.")
    p.add_argument("--chunk-size", type=int, default=512, help="rows per provider.complete() call")
    p.add_argument("--bad-response-log", default=None,
                   help="local JSONL path for rows dropped due to provider/format failures. "
                        "Default: {output}.bad_responses.jsonl")
    p.add_argument("--cache-from", action="append", default=[],
                   help="path(s) to existing *_explained.parquet to reuse explanations from "
                        "(joins on detokenized_text_truncated — same tokenizer + corpus slice "
                        "→ cache hit even across different base models). Repeat flag for "
                        "multiple sources. Storage backend: --cache-storage-cls.")
    p.add_argument("--cache-storage-cls", default="nla.datagen.storage.LocalStorage",
                   help="storage backend for --cache-from paths (may differ from main "
                        "--storage-cls, e.g. cache on cloud storage, output local)")
    add_storage_args(p)
    args = p.parse_args()

    assert "{text}" in args.instruction_template, "instruction-template must contain {text} placeholder"

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)
    provider: CompletionProvider = load_class(args.provider_cls)(**parse_kwargs(args.provider_kwargs))
    cache = load_explanation_cache(args.cache_from, args.cache_storage_cls) if args.cache_from else {}

    table = pq.read_table(storage.open_read(args.input))
    out_schema = table.schema.append(pa.field("api_explanation", pa.string()))
    storage.ensure_parent(args.output)
    bad_response_log = Path(args.bad_response_log or f"{args.output}.bad_responses.jsonl")
    bad_response_log.parent.mkdir(parents=True, exist_ok=True)

    # Per-chunk files for crash-safe resumption. Local-only (not via storage
    # backend — these are temp files). Existing chunk files are skipped on
    # restart; the API is never called twice for the same chunk.
    chunks_dir = Path(f"{args.output}.chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    def _log_bad_response(record: dict) -> None:
        with bad_response_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _process_chunk(chunk: pa.Table, chunk_start: int) -> tuple[pa.Table, int]:
        texts = chunk.column("detokenized_text_truncated").to_pylist()
        doc_ids = chunk.column("doc_id").to_pylist() if "doc_id" in chunk.column_names else [None] * len(texts)
        cached_expls = [lookup(cache, t) for t in texts]
        miss_idx = [i for i, e in enumerate(cached_expls) if e is None]
        miss_prompts = [args.instruction_template.format(text=texts[i]) for i in miss_idx]
        raw_completions = provider.complete(miss_prompts) if miss_prompts else []
        assert len(raw_completions) == len(miss_prompts), (
            f"provider returned {len(raw_completions)} completions for {len(miss_prompts)} prompts — "
            f"length mismatch violates the CompletionProvider contract"
        )
        miss_cleaned: dict[int, str | None] = {}
        for j, raw in zip(miss_idx, raw_completions, strict=True):
            # None = provider gave up on this prompt after exhausting retries.
            # Drop it (same path as failed-extract-pattern below).
            if raw is None:
                miss_cleaned[j] = None
                _log_bad_response({
                    "chunk_start": chunk_start,
                    "row_in_chunk": j,
                    "global_row": chunk_start + j,
                    "doc_id": doc_ids[j],
                    "text_sha256": hashlib.sha256(texts[j].encode("utf-8")).hexdigest(),
                    "reason": "provider_returned_none",
                    "raw_response": None,
                    "cleaned": None,
                    "feature_count": 0,
                })
                continue
            assert isinstance(raw, str) and raw, (
                f"provider returned bad completion at miss index {j}: {raw!r}. "
                f"CompletionProvider.complete() must return str or None."
            )
            cleaned = _extract_and_clean(raw, args.response_extract_pattern)
            miss_cleaned[j] = cleaned
            feature_count = 0 if cleaned is None else cleaned.count("\n\n") + 1
            if cleaned is None or feature_count < _MIN_FEATURES:
                _log_bad_response({
                    "chunk_start": chunk_start,
                    "row_in_chunk": j,
                    "global_row": chunk_start + j,
                    "doc_id": doc_ids[j],
                    "text_sha256": hashlib.sha256(texts[j].encode("utf-8")).hexdigest(),
                    "reason": "extract_failed" if cleaned is None else "too_few_features",
                    "raw_response": raw,
                    "cleaned": cleaned,
                    "feature_count": feature_count,
                })

        dropped = 0
        keep_mask: list[bool] = []
        explanations: list[str] = []
        for i, hit in enumerate(cached_expls):
            cleaned = hit if hit is not None else miss_cleaned[i]
            if cleaned is None or cleaned.count("\n\n") + 1 < _MIN_FEATURES:
                dropped += 1
                keep_mask.append(False)
                continue
            keep_mask.append(True)
            explanations.append(cleaned)
        if not all(keep_mask):
            chunk = chunk.filter(pa.array(keep_mask, type=pa.bool_()))
        return chunk.append_column("api_explanation", pa.array(explanations, type=pa.string())), dropped

    dropped_count = 0
    chunk_paths: list[Path] = []
    chunk_starts = list(range(0, table.num_rows, args.chunk_size))
    skipped = 0
    for chunk_start in tqdm(chunk_starts, desc="chunks"):
        chunk_path = chunks_dir / f"chunk_{chunk_start:08d}.parquet"
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            skipped += 1
            continue
        chunk_out, dropped = _process_chunk(table.slice(chunk_start, args.chunk_size), chunk_start)
        dropped_count += dropped
        if dropped:
            print(f"  chunk {chunk_start}: dropped {dropped} rows; see {bad_response_log}", flush=True)
        # tmp+rename: no partial chunk file if the process dies mid-write
        tmp = chunk_path.with_suffix(".tmp")
        pq.write_table(chunk_out, tmp)
        tmp.rename(chunk_path)
    if skipped:
        print(f"  resumed: skipped {skipped}/{len(chunk_starts)} already-completed chunks")

    # Merge chunks into final output via ParquetWriter (stream, not concat —
    # 100k-scale tables don't all fit in memory at once).
    row_count = 0
    with pq.ParquetWriter(storage.open_write(args.output), out_schema) as writer:
        for p in chunk_paths:
            t = pq.read_table(p)
            writer.write_table(t)
            row_count += t.num_rows

    # Record provider config in sidecar. Pull model/max_tokens/temperature via
    # getattr — providers aren't required to have these, but the default does.
    api_meta = NLAApiSummaryMeta(
        model=getattr(provider, "model", args.provider_cls),
        max_tokens=getattr(provider, "max_tokens", -1),
        temperature=getattr(provider, "temperature", -1.0),
        instruction_prompt=args.instruction_template,
    )
    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__explained",
        row_count=row_count,
        api_summaries=api_meta,
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.stage2_api_explain",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    assert row_count > 0, (
        f"ALL {dropped_count} rows dropped — either responses didn't match "
        f"--response-extract-pattern={args.response_extract_pattern!r} (truncated? "
        f"wrong tag?), or had fewer than {_MIN_FEATURES} features after cleanup. "
        f"Try: increase max_tokens, shorten the instruction, or check the tag matches "
        f"what your prompt asks for."
    )
    print(f"wrote {row_count} rows → {args.output}")
    if dropped_count > 0:
        print(f"  DROPPED {dropped_count} rows (response didn't match extract pattern)")


if __name__ == "__main__":
    main()
