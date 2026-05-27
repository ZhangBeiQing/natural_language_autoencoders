# input_embeds chunked-prefill shape mismatch

## Symptom

```
File "sglang/srt/mem_cache/memory_pool.py", line 894, in set_kv_buffer
    self.k_buffer[layer_id - self.start_layer][loc] = cache_k
RuntimeError: shape mismatch: value tensor of shape [8250, 4, 128]
  cannot be broadcast to indexing result of shape [8192, 4, 128]
```

## Relationship to PR #14110

**This is a different bug** with the same crash signature but **opposite polarity**:

| Scenario | cache_k shape | loc shape | Polarity |
|---|---|---|---|
| PR #14110 (retraction) | `len(embeds)` (smaller) | `len(embeds)+len(output_ids)` (bigger) | `cache_k < loc` |
| **This bug** (chunked prefill) | `len(embeds)` full (bigger) | truncated `extend_input_len` (smaller) | `cache_k > loc` |

If PR #14110's fix is applied and the crash still occurs with `cache_k > loc`, this is the cause.

## Root cause

`PrefillAdder.add_one_req_ignore_eos` truncates `fill_ids` and `extend_input_len` when a request would overflow `rem_chunk_tokens`, but does **not** truncate `req.input_embeds`. `ScheduleBatch.prepare_for_extend` then appends the full-length embed array:

```python
# schedule_batch.py prepare_for_extend
if req.input_embeds is not None:
    input_embeds.append(req.input_embeds)   # ← full array, not sliced
```

Result: model forward processes `len(full_embeds)` tokens but `out_cache_loc` was sized for the truncated `extend_input_len`.

## Trigger conditions

Rare in normal operation (requests typically arrive one per scheduler iteration). Requires a **thundering herd** — many input_embeds requests queued in a single iteration. In our RL setup this happens at rollout start immediately after `update_weights_from_distributed`:

- 256 requests per engine arrive near-simultaneously (1024 rollout requests ÷ 4 engines)
- With 125-token prompts: 65 requests × 125 = 8125 tokens, request #66 has `rem_chunk_tokens = 8192−8125 = 67`
- Adder truncates request #66's `fill_ids` to 67 but `input_embeds` stays at 125
- Overflow = 125 − 67 = **58 tokens** exactly

The delta in the traceback is `prompt_len − (chunked_prefill_size mod prompt_len)`.

## Log evidence (from our crash at RL step 1915)

- KV `token usage: 0.00` throughout — **no memory pressure, no retraction occurred**
- Normal scheduler cadence: `Prefill batch, #new-seq: 1, #new-token: 125` (requests trickle in one-by-one)
- Crash timing: immediately after `update_weights` at rollout start (burst of requests)
- The crashing batch's `#new-seq: 66` log line never emitted (SIGQUIT before stdout flush)

## Fix

Slice `input_embeds` with the same bounds the adder already applied to `fill_ids`:

```python
# schedule_batch.py prepare_for_extend
if req.input_embeds is not None:
    # pre_len = len(prefix_indices) — rows already consumed in prior chunks
    # req.extend_input_len — this chunk's budget (truncated by PrefillAdder)
    # Non-chunked case: pre_len=0, extend_input_len=len(embeds) → full slice (no-op)
    input_embeds.append(
        req.input_embeds[pre_len : pre_len + req.extend_input_len]
    )
```

Both `pre_len` and `req.extend_input_len` are already in scope from the enclosing loop. The slice is a no-op for non-chunked requests and correctly handles chunk continuation.

## Config workaround (no code change)

Set `prefill_max_requests = ⌊chunked_prefill_size / prompt_len⌋ − 1` (e.g. 64 for 8192/125). Caps the batch below the chunking boundary. Zero throughput cost when requests arrive one-at-a-time anyway. Fragile to prompt length changes.

## Our patch

`nla_sglang_input_embeds_chunked_prefill_fix.patch` — apply manually before launching SGLang. Keep the #14110 retract fix too; it's correct for its case, just wasn't *this* case.
