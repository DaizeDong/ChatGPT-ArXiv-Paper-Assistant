"""The ONE place this repo talks to a language model.

Backends, in resolution order:

``llmcall``  a standalone unified call primitive (chain: codexg -> codex -> cc ->
             claude). Used when the package imports. No API key: the chain runs
             through locally installed CLIs.
``agent``    this repo's own ``claude -p`` transport (utils/agent_runner). The
             self-sufficiency floor: a bare clone with the ``claude`` CLI works
             with nothing else installed.
``openai``   the historical HTTP path. Only reachable when explicitly selected
             AND a key is present. It is no longer a default anywhere.

WHY DETECTION IS AN IMPORT, NOT AN ENV VAR OR A VENDORED COPY. A guarded import
asks the question at the moment it matters and answers it from the interpreter
that will actually make the call. An environment variable answers it from
whatever the process happened to inherit, which is not reliable under schedulers
and service managers; a vendored copy answers it from a snapshot that starts
drifting the day it is taken. A clone without llmcall degrades to ``agent`` and
still runs; nothing in the repo imports llmcall at module scope, and
requirements.txt gains no dependency.

THE CALL LEDGER IS THE POINT, NOT A SIDE FEATURE. Health used to be inferred from
OpenAI token counts, which silently stops working the moment the backend stops
reporting tokens -- an outage detector that reads zero tokens as "the model never
ran" would fire on every single llmcall run forever. So every call, on every
backend, is recorded here as attempted/succeeded. "We tried N and none worked" is
the provider-agnostic statement of an outage, and it is what
utils/pipeline_health reads.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from arxiv_assistant.utils.agent_runner import AgentError, run_agent

BACKEND_LLMCALL = "llmcall"
BACKEND_AGENT = "agent"
BACKEND_OPENAI = "openai"

#: Env override, mainly for tests and for pinning a backend on a deploy host.
BACKEND_ENV = "ARXIV_ASSISTANT_LLM_BACKEND"

DEFAULT_TIMEOUT_S = 180.0
#: Judgment tasks get the strongest reasoning tier. Relevance, novelty and delta
#: scoring are judgments, not formatting, so they are never downgraded for cost.
DEFAULT_EFFORT = "max"


@dataclass
class CallLedger:
    """Per-process record of every model call this run attempted.

    Read by pipeline_health. Kept deliberately small: counts, the backends and
    providers that actually answered, and a capped list of error strings.
    """

    attempted: int = 0
    succeeded: int = 0
    by_backend: Dict[str, int] = field(default_factory=dict)
    by_provider: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    #: Per-ATTEMPT lines from llmcall's own log hook. The chain tries several
    #: providers per call, so "which provider answered" is not the whole story:
    #: this is where you see that codexg was rung first and timed out, that the
    #: gateway substituted its strongest model, and how long each leg took.
    #: Without it the ledger can say a call succeeded but not what actually ran.
    trace: List[str] = field(default_factory=list)
    #: Wall-clock seconds spent inside model calls. This transport is not billed
    #: per token and reports none, so "what did this run cost" is answered in
    #: calls and seconds in calls and seconds, not in dollars.
    #: Reporting a fabricated $0.00 would be worse than reporting nothing.
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    MAX_ERRORS = 20
    MAX_TRACE = 200

    def record_attempt(self, backend: str) -> None:
        with self._lock:
            self.attempted += 1
            self.by_backend[backend] = self.by_backend.get(backend, 0) + 1

    def record_success(self, provider: str) -> None:
        with self._lock:
            self.succeeded += 1
            key = provider or "unknown"
            self.by_provider[key] = self.by_provider.get(key, 0) + 1

    def record_seconds(self, elapsed: float) -> None:
        with self._lock:
            self.seconds += max(0.0, float(elapsed))

    def record_error(self, message: str) -> None:
        with self._lock:
            if len(self.errors) < self.MAX_ERRORS:
                self.errors.append(str(message)[:300])

    def record_trace(self, message: str) -> None:
        """One line from the backend's own per-attempt log."""
        with self._lock:
            if len(self.trace) < self.MAX_TRACE:
                self.trace.append(str(message)[:300])

    def answering_providers(self) -> List[str]:
        """Providers that actually produced an answer, most used first."""
        return [p for p, _ in sorted(self.by_provider.items(), key=lambda kv: -kv[1])]

    @property
    def failed(self) -> int:
        return max(0, self.attempted - self.succeeded)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "by_backend": dict(self.by_backend),
            "by_provider": dict(self.by_provider),
            "errors": list(self.errors),
            "seconds": round(self.seconds, 1),
            "trace": list(self.trace),
        }

    def reset(self) -> None:
        with self._lock:
            self.attempted = 0
            self.succeeded = 0
            self.by_backend.clear()
            self.by_provider.clear()
            self.errors.clear()
            self.trace.clear()
            self.seconds = 0.0


LEDGER = CallLedger()


@dataclass(frozen=True)
class GatewayResult:
    text: str
    data: Any = None
    backend: str = ""
    provider: str = ""

    def __bool__(self) -> bool:
        return bool(self.text) or self.data is not None


def llmcall_available() -> bool:
    """True when the llmcall primitive is importable on this machine."""
    try:
        import llmcall  # noqa: F401
    except Exception:
        return False
    return True


def openai_available() -> bool:
    from arxiv_assistant.utils.llm_client import resolve_openai_config

    key, _ = resolve_openai_config()
    return bool(key)


def available_backends() -> List[str]:
    backends = []
    if llmcall_available():
        backends.append(BACKEND_LLMCALL)
    backends.append(BACKEND_AGENT)  # always available in principle (claude CLI)
    if openai_available():
        backends.append(BACKEND_OPENAI)
    return backends


def resolve_backend(config: Any = None) -> str:
    """Pick the backend: env override, then ``[LLM] backend``, then detection.

    ``auto`` (the default) prefers llmcall and falls back to the repo-local agent
    transport. It never falls back to OpenAI: a dead key must surface as an
    outage, not be silently reintroduced as the default.
    """
    override = (os.environ.get(BACKEND_ENV) or "").strip().lower()
    if not override and config is not None:
        try:
            if config.has_section("LLM"):
                override = (config["LLM"].get("backend", "") or "").strip().lower()
        except (AttributeError, KeyError, TypeError):
            override = ""

    if override in (BACKEND_LLMCALL, BACKEND_AGENT, BACKEND_OPENAI):
        return override
    if override and override != "auto":
        raise ValueError(
            f"Unknown LLM backend {override!r}. Valid: llmcall, agent, openai, auto."
        )
    return BACKEND_LLMCALL if llmcall_available() else BACKEND_AGENT


def _effort(config: Any) -> str:
    if config is None:
        return DEFAULT_EFFORT
    try:
        if config.has_section("LLM"):
            return (config["LLM"].get("effort", "") or "").strip() or DEFAULT_EFFORT
    except (AttributeError, KeyError, TypeError):
        pass
    return DEFAULT_EFFORT


def call(
    prompt: str,
    *,
    schema: Optional[dict] = None,
    config: Any = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    tools: Optional[List[str]] = None,
    llmcall_fn: Optional[Callable[..., Any]] = None,
    agent_fn: Optional[Callable[..., dict]] = None,
) -> GatewayResult:
    """Send one prompt through the resolved backend.

    Raises :class:`AgentError` on failure, which is the exception every existing
    caller in this repo already catches and degrades on. Reusing it means the
    gateway can be dropped under those callers without changing their error
    handling.

    ``schema`` is the same minimal JSON-Schema shape ``agent_runner`` validates:
    ``{"required": [...], "properties": {...}}``. Both backends enforce it.
    """
    chosen = backend or resolve_backend(config)
    LEDGER.record_attempt(chosen)
    started = time.monotonic()
    try:
        if chosen == BACKEND_LLMCALL:
            return _call_llmcall(
                prompt, schema=schema, config=config, model=model,
                timeout_s=timeout_s, llmcall_fn=llmcall_fn,
            )
        if chosen == BACKEND_AGENT:
            return _call_agent(
                prompt, schema=schema, config=config, model=model,
                timeout_s=timeout_s, tools=tools, agent_fn=agent_fn,
            )
        if chosen == BACKEND_OPENAI:
            return _call_openai(prompt, config=config, model=model, timeout_s=timeout_s)
        raise AgentError(f"Unknown LLM backend: {chosen!r}")
    finally:
        # Recorded for failures too: a chain that burns three minutes before
        # giving up costs real wall clock, and a cost report that only counts
        # successes would hide exactly the runs worth noticing.
        LEDGER.record_seconds(time.monotonic() - started)


def _call_llmcall(prompt, *, schema, config, model, timeout_s, llmcall_fn):
    if llmcall_fn is None:
        try:
            from llmcall import call as llmcall_fn  # type: ignore[no-redef]
        except Exception as exc:  # noqa: BLE001 - absence is a normal degrade
            raise AgentError(f"llmcall backend selected but not importable: {exc}") from exc

    kwargs: Dict[str, Any] = {
        "mode": "judge",       # read-only, MCP off, deterministic
        "timeout": float(timeout_s),
        # Effort is a cross-provider policy and is passed EVERY time rather than
        # left to each provider's own config, which drifts. Judgment tasks take
        # the top tier; see DEFAULT_EFFORT.
        "effort": _effort(config),
        # llmcall reports each leg of the chain through this hook: which provider
        # was rung, whether the gateway substituted its strongest model, how long
        # each attempt took. That is the only place "which model actually ran" is
        # observable, so it goes into the ledger instead of the void.
        "log": LEDGER.record_trace,
    }
    if schema is not None:
        kwargs["schema"] = schema
    if model:
        kwargs["model"] = model

    result = llmcall_fn(prompt, **kwargs)

    # llmcall never raises; a falsy Result means the whole chain failed.
    if not result:
        error = getattr(result, "error", "") or "llmcall chain returned nothing"
        LEDGER.record_error(str(error))
        raise AgentError(f"llmcall chain failed: {error}")

    provider = str(getattr(result, "provider", "") or "")
    LEDGER.record_success(provider)
    return GatewayResult(
        text=str(getattr(result, "text", "") or ""),
        data=getattr(result, "data", None),
        backend=BACKEND_LLMCALL,
        provider=provider,
    )


def _call_agent(prompt, *, schema, config, model, timeout_s, tools, agent_fn):
    from arxiv_assistant.utils.llm_client import resolve_agent_model

    runner = agent_fn or run_agent
    resolved = model or resolve_agent_model(config) if config is not None else (model or "")
    if not resolved:
        resolved = model or "claude-sonnet-5"

    # run_agent always validates against a schema; give it a permissive one when
    # the caller wants raw text back.
    effective_schema = schema if schema is not None else {"required": [], "properties": {}}
    try:
        payload = runner(
            prompt, schema=effective_schema, model=resolved,
            tools=tools, timeout_s=int(timeout_s),
        )
    except AgentError as exc:
        LEDGER.record_error(str(exc))
        raise

    LEDGER.record_success(BACKEND_AGENT)
    return GatewayResult(
        text=payload.get("text", "") if isinstance(payload, dict) else "",
        data=payload,
        backend=BACKEND_AGENT,
        provider="claude",
    )


def _call_openai(prompt, *, config, model, timeout_s):
    from arxiv_assistant.utils.llm_client import get_openai_client, resolve_llm_model

    client = get_openai_client()
    resolved = model or resolve_llm_model(config)
    try:
        completion = client.chat.completions.create(
            model=resolved,
            messages=[{"role": "user", "content": prompt}],
            seed=0,
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - normalised into AgentError below
        LEDGER.record_error(str(exc))
        raise AgentError(f"OpenAI call failed: {exc}") from exc

    LEDGER.record_success("openai")
    return GatewayResult(
        text=completion.choices[0].message.content or "",
        data=completion,
        backend=BACKEND_OPENAI,
        provider="openai",
    )


def legacy_openai_key_missing(api_key: Any, config: Any) -> bool:
    """True when the legacy OpenAI backend was requested and its key is unusable.

    Lives here rather than in ``environment`` because it is a question about the
    backend policy, and because ``environment`` cannot be imported in a test: it
    reads config by relative path, fetches a live RSS feed and creates dated
    output directories, all at import time.

    An EMPTY string counts as missing. ``is None`` alone would treat
    ``OPENAI_API_KEY=`` -- the shape an unset CI secret expands to -- as a
    present key and defer the failure to a 401 on the first batch. That
    misattribution, a configuration fault surfacing as a model error, is how the
    last outage stayed invisible for three months.
    """
    if str(api_key or "").strip():
        return False
    try:
        backend = (
            config["LLM"].get("backend", "auto").strip().lower()
            if config.has_section("LLM")
            else "auto"
        )
    except Exception:  # noqa: BLE001 - see below; failing open is correct HERE
        # Deliberately broad, and deliberately failing open. This predicate only
        # decides whether to BLOCK STARTUP. If the config cannot be read we let
        # the process start: the call ledger still records that every model call
        # failed, so a genuinely misconfigured legacy backend surfaces as an
        # outage a few seconds later, through the mechanism built for it. Failing
        # closed here would instead turn an unreadable config into an import
        # error with no diagnosis at all.
        backend = "auto"
    return backend == BACKEND_OPENAI


def describe_backend(config: Any = None) -> str:
    """One line for logs and for the digest header."""
    try:
        chosen = resolve_backend(config)
    except ValueError as exc:
        return f"LLM backend: INVALID ({exc})"
    detail = ""
    if chosen == BACKEND_LLMCALL:
        try:
            from llmcall import active_chain

            detail = f" chain={'->'.join(active_chain())}"
        except Exception:  # noqa: BLE001 - cosmetic only
            detail = ""
    return f"LLM backend: {chosen}{detail}"
