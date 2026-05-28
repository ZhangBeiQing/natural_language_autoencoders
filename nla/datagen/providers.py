"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path

import anthropic

_DOTENV_LOADED = False


def _load_dotenv_if_present() -> None:
    """Load simple KEY=VALUE entries from .env without adding a dependency."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    candidates = []
    explicit = os.environ.get("NLA_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit))
    cwd = Path.cwd()
    candidates.extend([cwd / ".env", *(p / ".env" for p in cwd.parents)])

    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                os.environ.setdefault(key, value)
        return


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        # refusal: source text tripped safety — no answer coming, drop this row.
        # content may be [] or the refusal message; either way, no explanation.
        if resp.stop_reason == "refusal":
            return None
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion — refusing to emit blank explanation"
        return text

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out


class OpenAIChatProvider(CompletionProvider):
    """OpenAI-compatible chat completions provider.

    Use this for providers that expose `/chat/completions`, including DeepSeek
    and MiMo token-plan endpoints. Provider-specific subclasses set defaults
    for model name, base URL, and environment variable names.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        api_base_env: str = "OPENAI_API_BASE",
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
        timeout: float = 120.0,
        extra_body: dict | None = None,
        abort_on_error: bool = True,
        rpm: int | None = None,
    ):
        from openai import AsyncOpenAI

        _load_dotenv_if_present()
        key = api_key or os.environ.get(api_key_env, "")
        assert key, f"{api_key_env} is required for {type(self).__name__}"
        base = api_base or os.environ.get(api_base_env, "")
        assert base, f"{api_base_env} is required for {type(self).__name__}"
        self.client = AsyncOpenAI(
            api_key=key,
            base_url=base,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.extra_body = extra_body or {}
        self.abort_on_error = abort_on_error
        self.rpm = rpm
        self._min_request_interval = 0.0 if rpm is None or rpm <= 0 else 60.0 / rpm
        self._rate_lock: asyncio.Lock | None = None
        self._next_request_at = 0.0

    async def _throttle(self) -> None:
        if self._min_request_interval <= 0:
            return
        if self._rate_lock is None:
            self._rate_lock = asyncio.Lock()
        async with self._rate_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_request_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_request_at = max(now, self._next_request_at) + self._min_request_interval

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            await self._throttle()
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=self.extra_body or None,
            )
        choice = resp.choices[0]
        text = choice.message.content
        if text is None:
            return None
        text = text.strip()
        return text or None

    def complete(self, prompts: list[str]) -> list[str | None]:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        tolerated = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)

        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, tolerated):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if self.abort_on_error and n_failed:
            failed_types = sorted({type(r).__name__ for r in raw if isinstance(r, tolerated)})
            raise RuntimeError(
                f"{type(self).__name__} got {n_failed}/{len(prompts)} retry-exhausted API errors "
                f"({', '.join(failed_types)}). Aborting this Stage 2 chunk without writing it; "
                "completed chunk files remain resumable. Refill quota or fix the transient issue, "
                "then rerun the same Stage 2 command."
            )
        if n_failed or n_refused:
            print(
                f"  [{type(self).__name__}] dropped "
                f"{n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}"
            )
        return out


class DeepSeekProvider(OpenAIChatProvider):
    """DeepSeek API provider using the OpenAI SDK.

    Key: thinking mode is DISABLED by default for this provider, because
    NLA explanations are short (~100 words) and thinking tokens would be
    wasted cost (they're billed but not used). Enable with thinking=True
    if you want reasoning before the explanation.

    Usage:
        --provider-cls nla.datagen.providers.DeepSeekProvider
        --provider-kwargs '{"model": "deepseek-v4-pro", "concurrency": 50}'

    Environment variables:
        DEEPSEEK_API_KEY: required
        DEEPSEEK_API_BASE: optional, defaults to https://api.deepseek.com
    """

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
        thinking: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        abort_on_error: bool = True,
        rpm: int | None = None,
    ):
        extra_body = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        super().__init__(
            model=model,
            api_key_env="DEEPSEEK_API_KEY",
            api_base_env="DEEPSEEK_API_BASE",
            api_key=api_key,
            api_base=api_base or "https://api.deepseek.com",
            max_tokens=max_tokens + 4096 if thinking else max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            max_retries=max_retries,
            timeout=timeout,
            abort_on_error=abort_on_error,
            rpm=rpm,
            extra_body=extra_body,
        )


class MiMoProvider(OpenAIChatProvider):
    """Xiaomi MiMo token-plan provider using the OpenAI SDK."""

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 50,
        max_retries: int = 10,
        thinking: bool = False,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 120.0,
        abort_on_error: bool = True,
        rpm: int | None = None,
    ):
        extra_body = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        super().__init__(
            model=model,
            api_key_env="MIMO_API_KEY",
            api_base_env="MIMO_API_BASE",
            api_key=api_key,
            api_base=api_base or "https://token-plan-cn.xiaomimimo.com/v1",
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            max_retries=max_retries,
            timeout=timeout,
            abort_on_error=abort_on_error,
            rpm=rpm,
            extra_body=extra_body,
        )
