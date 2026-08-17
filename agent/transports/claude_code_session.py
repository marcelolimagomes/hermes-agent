"""Claude Code stream-json session client.

Drives the first-party `claude` CLI in programmatic mode over stdio, the same
shape `codex_app_server.py` uses for the Codex CLI. Transport is
newline-delimited JSON: spawn `claude --print --input-format stream-json
--output-format stream-json`, register a `PreToolUse` hook in the `initialize`
control request, then send user messages and consume the event stream.

Why a CLI transport and not the Anthropic provider: measured on the appliance
host, the Anthropic API refuses Hermes as a third-party app (`HTTP 400`, extra
usage) even with a valid Claude Code credential, while the CLI works on the
subscription plan. Registration of this provider is authorized by ADR-020 of
the taskblu-co-working-assistant appliance; it is explicitly *registered, not
promoted* — selecting it for a lane is a separate decision.

This module is the wire-level speaker plus turn projection. Every behaviour
below was measured against `claude` 2.1.204; the comments name what was
measured and, where a previous implementation got it wrong, what it got wrong.

Security boundary, stated plainly: the `PreToolUse` hook closes the four
observed bypasses of `can_use_tool` (host `permissions.allow` rules,
`--allowedTools`, `--disallowedTools` auto-deny, and the CLI's internal
read-only classifier). It is NOT a sandbox: it arrives over the same stdio
transport, in the same process, and when the supervising client dies the CLI
abandons the hook and degrades to its own base policy. Unlike the Codex
runtime, there is no `sandbox_mode` equivalent.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tools.environments.local import hermes_subprocess_env

# Minimum CLI version this transport was measured against. `claude --version`
# prints "2.1.204 (Claude Code)" — the product suffix comes AFTER the semver,
# unlike `codex --version` which prints "codex-cli 0.146.0". Compare by
# extracted semver, never by string equality against a fixed prefix.
MIN_CLAUDE_VERSION = (2, 1, 204)

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")

# Argv substrings that remove the permission barrier. Listed here ONLY so the
# transport can refuse them: `build_argv` raises before spawning if any appear.
# `--allow-dangerously-skip-permissions` is a distinct flag from
# `--dangerously-skip-permissions`; a deny-list matching only the latter has a
# hole, which is why this matches by substring over the assembled argv.
BANNED_ARGV_SUBSTRINGS = (
    "dangerously-skip-permissions",
    "bypassPermissions",
)


class ClaudeCodeSessionError(RuntimeError):
    """Transport-level failure that the caller must surface, never swallow."""


class ClaudeCodeIdentityError(ValueError):
    """Argv combination the CLI rejects, refused before the spawn.

    Measured: `--session-id` with `--resume` and without `--fork-session` exits
    1 with `--session-id can only be used with --continue or --resume if
    --fork-session is also specified.` Catching it here turns an opaque exit
    code into a named error.
    """


@dataclass(frozen=True)
class Identity:
    """The three argv forms, which are NOT interchangeable.

    Measured against the CLI; there is no single "lane argv":

      new    : --session-id <uuid>            -> init emits exactly that id
      resume : --resume <id>                  -> init emits the ORIGINAL id
      fork   : --resume <id> --session-id <new> --fork-session
    """

    form: str
    session_id: Optional[str] = None
    parent_session_id: Optional[str] = None

    @staticmethod
    def new(session_id: Optional[str] = None) -> "Identity":
        return Identity(form="new", session_id=session_id or str(uuid.uuid4()))

    @staticmethod
    def resume(parent_session_id: str) -> "Identity":
        return Identity(form="resume", parent_session_id=parent_session_id)

    @staticmethod
    def fork(parent_session_id: str, session_id: Optional[str] = None) -> "Identity":
        return Identity(
            form="fork",
            parent_session_id=parent_session_id,
            session_id=session_id or str(uuid.uuid4()),
        )

    def validate(self) -> None:
        if self.form not in {"new", "resume", "fork"}:
            raise ClaudeCodeIdentityError(f"unknown identity form: {self.form!r}")
        if self.form == "new" and not self.session_id:
            raise ClaudeCodeIdentityError("form=new requires a session_id")
        if self.form in {"resume", "fork"} and not self.parent_session_id:
            raise ClaudeCodeIdentityError(f"form={self.form} requires parent_session_id")
        if self.form == "resume" and self.session_id:
            raise ClaudeCodeIdentityError(
                "form=resume must not fix --session-id; use form=fork to branch"
            )

    def argv_fragment(self) -> list[str]:
        self.validate()
        if self.form == "new":
            return ["--session-id", str(self.session_id)]
        if self.form == "resume":
            return ["--resume", str(self.parent_session_id)]
        return [
            "--resume",
            str(self.parent_session_id),
            "--session-id",
            str(self.session_id),
            "--fork-session",
        ]


@dataclass
class GateDecision:
    """Verdict for one tool request. Default is deny; allow is deliberate."""

    allow: bool = False
    reason: str = "no policy decision"


@dataclass
class ToolCall:
    """One tool the model asked for, and whether the gate actually saw it.

    `gated=False` means the tool ran WITHOUT the supervisor being consulted —
    the bypass paths. Recording it is what makes an audit possible; a transport
    that only records denials cannot tell "allowed" from "never asked".
    """

    tool_use_id: str
    tool_name: str
    gated: bool = False
    decision: Optional[str] = None


@dataclass
class ClaudeTurnResult:
    """Projection of one turn into the shape Hermes consumes."""

    session_id: Optional[str] = None
    final_text: Optional[str] = None
    status: str = "unknown"
    is_error: bool = False
    subtype: Optional[str] = None
    terminal_reason: Optional[str] = None
    interrupted: bool = False
    num_turns: Optional[int] = None
    total_cost_usd: Optional[float] = None
    usage: dict = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    permission_denials: list[dict] = field(default_factory=list)
    projection_failures: list[dict] = field(default_factory=list)
    blocking_anomalies: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    @property
    def gate_bypasses(self) -> list[ToolCall]:
        return [c for c in self.tool_calls if not c.gated]


def parse_claude_version(output: str) -> Optional[tuple[int, int, int]]:
    """Extract semver from `claude --version`.

    Measured output: "2.1.204 (Claude Code)". Equality against a composed
    string fails because the product name trails the version, so extract.
    """
    match = _VERSION_RE.search(output or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def build_argv(
    claude_bin: str,
    identity: Identity,
    *,
    tools: str = "Bash",
    model: Optional[str] = None,
    mcp_config: Optional[str] = None,
    setting_sources: str = "",
    max_budget_usd: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Assemble the lane argv, refusing barrier-removing flags.

    `tools` must not be empty: measured, `--tools ""` yields a `system/init`
    with `tools=[]`, which makes the gate unreachable because the model has
    nothing to request. An empty surface looks safe and is actually untestable.
    """
    if not tools:
        raise ClaudeCodeIdentityError(
            "empty tool surface makes the approval gate unreachable; pass a tool set"
        )
    argv = [
        claude_bin,
        "--print",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        # Measured: stream-json output without --verbose exits 1.
        "--verbose",
    ]
    argv += identity.argv_fragment()
    argv += [
        # Isolation from the operator's host settings. Measured: this drops the
        # tool surface from 68 to 26 and removes host hooks, plugins and MCP.
        "--setting-sources",
        setting_sources,
        "--strict-mcp-config",
        # The stdio permission channel. Without it the headless mode auto-denies
        # and the client is blind: no control_request ever arrives.
        "--permission-prompt-tool",
        "stdio",
        "--tools",
        tools,
        # Deliberately empty: every call must climb to the supervisor rather
        # than being auto-allowed by a flag.
        "--allowedTools",
        "",
    ]
    if mcp_config:
        argv += ["--mcp-config", mcp_config]
    if max_budget_usd:
        argv += ["--max-budget-usd", str(max_budget_usd)]
    if model:
        argv += ["--model", model]
    argv += list(extra_args or [])

    for item in argv:
        for banned in BANNED_ARGV_SUBSTRINGS:
            if banned in item:
                raise ClaudeCodeIdentityError(
                    f"refusing argv that removes the permission barrier: {item!r}"
                )
    return argv


def default_policy(tool_name: Optional[str], tool_input: Any, channel: str) -> GateDecision:
    """Fail-closed default. A transport that defaults to allow is not a gate."""
    return GateDecision(allow=False, reason=f"no policy configured for {tool_name!r} via {channel}")


class ClaudeCodeSession:
    """Session-persistent stdio client for the Claude Code CLI.

    Threading model mirrors CodexAppServerClient: the caller drives turns
    synchronously; one reader thread parses stdout and dispatches. Not async,
    for the same reason stated there — AIAgent.run_conversation() is sync.

    Measured lifecycle facts that shape this class:
      - The process stays alive after a turn completes and accepts a second
        user message on the same session, preserving context. This is a
        session transport, not a process-per-turn one.
      - A turn starts on the newline; stdin does NOT need to be closed.
      - An unknown control_request subtype returns an error WITHOUT killing
        the process, so control errors are recoverable.
      - A malformed user message DOES kill the process, so serialization must
        never interleave: see the stdin lock.
    """

    def __init__(
        self,
        claude_bin: str = "claude",
        *,
        identity: Optional[Identity] = None,
        cwd: Optional[str] = None,
        tools: str = "Bash",
        model: Optional[str] = None,
        mcp_config: Optional[str] = None,
        max_budget_usd: Optional[str] = None,
        policy: Callable[[Optional[str], Any, str], GateDecision] = default_policy,
        register_hook: bool = True,
        gate_decision_timeout: float = 10.0,
        env: Optional[dict[str, str]] = None,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self._bin = claude_bin
        self._identity = identity or Identity.new()
        self._identity.validate()
        self._cwd = cwd
        self._policy = policy
        self._register_hook = register_hook
        self._gate_decision_timeout = gate_decision_timeout
        self._argv = build_argv(
            claude_bin,
            self._identity,
            tools=tools,
            model=model,
            mcp_config=mcp_config,
            max_budget_usd=max_budget_usd,
            extra_args=extra_args,
        )

        # Same centralized helper the Codex transport uses: provider creds flow,
        # Tier-1 Hermes secrets are stripped. A model-driving CLI has no use for
        # gateway tokens, dashboard sessions or side-LLM keys.
        spawn_env = hermes_subprocess_env(inherit_credentials=True)
        # Measured: inheriting these changes auth resolution and tool surface.
        for leaked in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
            spawn_env.pop(leaked, None)
        if env:
            spawn_env.update(env)
        self._env = spawn_env

        self._proc: Optional[subprocess.Popen] = None
        # send() is called from BOTH the caller thread (handshake, user message,
        # interrupt) and the reader thread (gate responses). Two interleaved
        # NDJSON lines do not cost an event: a malformed line KILLS the process.
        self._stdin_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._result_ev = threading.Event()
        self._events: "queue.Queue[dict]" = queue.Queue()
        self._stderr_tail: list[str] = []
        self._lines = 0
        self._reader_died: Optional[str] = None
        self._pending_gate: dict[str, dict] = {}
        self._session_id: Optional[str] = None
        self._result: Optional[ClaudeTurnResult] = None
        self._turn: Optional[ClaudeTurnResult] = None

    # --- process lifecycle -------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            raise ClaudeCodeSessionError("session already started")
        self._proc = subprocess.Popen(  # noqa: S603 — argv is built and vetted above
            self._argv,
            cwd=self._cwd,
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._reader, name="claude-stdout", daemon=True).start()
        threading.Thread(target=self._stderr_reader, name="claude-stderr", daemon=True).start()
        if self._register_hook:
            self._handshake()

    def close(self, timeout: float = 3.0) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stderr_tail(self, n: int = 20) -> list[str]:
        with self._state_lock:
            return list(self._stderr_tail[-n:])

    # --- wire --------------------------------------------------------------

    def send(self, obj: dict) -> bool:
        """Serialize and write one NDJSON line under a lock.

        The lock is not defensive style: without it two threads interleave a
        line and the CLI dies with a JSON parse error, taking the session and
        its context with it.
        """
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            return False
        try:
            line = json.dumps(obj, ensure_ascii=False) + "\n"
        except (TypeError, ValueError) as exc:
            # Serialization failure must not reach stdin as a partial write.
            raise ClaudeCodeSessionError(f"unserializable payload: {exc}") from exc
        with self._stdin_lock:
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return False
        return True

    def _handshake(self) -> None:
        """Register the PreToolUse hook in-band.

        Measured: a single `*` matcher fires for every tool, and the hook is
        invoked for the four paths that bypass `can_use_tool`. This is the only
        channel that reaches the CLI's internal read-only classifier, which no
        flag disables.
        """
        request: dict[str, Any] = {"subtype": "initialize"}
        request["hooks"] = {
            "PreToolUse": [{"matcher": "*", "hookCallbackIds": ["hermes_gate"]}]
        }
        self.send(
            {
                "type": "control_request",
                "request_id": f"hermes_init_{uuid.uuid4().hex[:8]}",
                "request": request,
            }
        )

    def interrupt(self) -> bool:
        return self.send(
            {
                "type": "control_request",
                "request_id": f"hermes_int_{uuid.uuid4().hex[:8]}",
                "request": {"subtype": "interrupt"},
            }
        )

    # --- reader ------------------------------------------------------------

    def _stderr_reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in proc.stderr:
            with self._state_lock:
                self._stderr_tail.append(raw.rstrip("\n"))
                del self._stderr_tail[:-200]

    def _reader(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for raw in proc.stdout:
                self._lines += 1
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    self._record_failure({"_raw_line": stripped[:4000]}, exc)
                    continue

                # Control dispatch and projection share ONE guard on purpose.
                # An earlier revision dispatched control OUTSIDE the try: a
                # control_request whose `request` field was a string killed this
                # thread silently — it is a daemon — and the turn ended with no
                # result at all. Protection must cover both branches.
                try:
                    kind = event.get("type")
                    if kind == "control_request":
                        self._on_control_request(event)
                        continue
                    if kind == "control_response":
                        continue
                    with self._state_lock:
                        self._project(event)
                except Exception as exc:  # noqa: BLE001 — deliberate
                    self._record_failure(event, exc)
                    continue

                if event.get("type") == "result":
                    self._result_ev.set()
        except Exception as exc:  # noqa: BLE001
            # A dead reader must be observable. Silence here is what turned a
            # crash into "the turn simply produced nothing".
            self._reader_died = f"{type(exc).__name__}: {exc}"
        finally:
            self._result_ev.set()

    def _record_failure(self, event: dict, exc: BaseException) -> None:
        with self._state_lock:
            turn = self._turn
            if turn is None:
                return
            turn.projection_failures.append(
                {
                    "line": self._lines,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_event": event,
                }
            )

    # --- gate --------------------------------------------------------------

    def _on_control_request(self, event: dict) -> None:
        request = event.get("request")
        if not isinstance(request, dict):
            raise ClaudeCodeSessionError(f"control_request without an object body: {request!r}")
        subtype = request.get("subtype")
        request_id = event.get("request_id")

        if subtype not in {"can_use_tool", "hook_callback"}:
            return

        if subtype == "hook_callback":
            payload = request.get("input") or {}
            tool_name = payload.get("tool_name")
            tool_input = payload.get("tool_input")
        else:
            tool_name = request.get("tool_name")
            tool_input = request.get("input")

        decision = self._safe_policy(tool_name, tool_input, subtype)
        self._note_gate(request.get("tool_use_id") or "", tool_name or "", decision)

        if subtype == "hook_callback":
            body = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow" if decision.allow else "deny",
                    "permissionDecisionReason": decision.reason,
                }
            }
        else:
            body = {"behavior": "allow" if decision.allow else "deny", "message": decision.reason}

        self.send(
            {
                "type": "control_response",
                "response": {"subtype": "success", "request_id": request_id, "response": body},
            }
        )

    def _safe_policy(self, tool_name, tool_input, channel: str) -> GateDecision:
        """A policy that raises or hangs must become a denial, never an allow."""
        try:
            decision = self._policy(tool_name, tool_input, channel)
        except Exception as exc:  # noqa: BLE001
            return GateDecision(allow=False, reason=f"policy raised: {type(exc).__name__}")
        if not isinstance(decision, GateDecision):
            return GateDecision(allow=False, reason="policy returned a non-decision")
        return decision

    def _note_gate(self, tool_use_id: str, tool_name: str, decision: GateDecision) -> None:
        with self._state_lock:
            turn = self._turn
            if turn is None:
                return
            for call in turn.tool_calls:
                if call.tool_use_id == tool_use_id:
                    call.gated = True
                    call.decision = "allow" if decision.allow else "deny"
                    break
            else:
                turn.tool_calls.append(
                    ToolCall(
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        gated=True,
                        decision="allow" if decision.allow else "deny",
                    )
                )
            if not decision.allow:
                turn.permission_denials.append(
                    {"tool_use_id": tool_use_id, "tool_name": tool_name, "reason": decision.reason}
                )

    # --- projection --------------------------------------------------------

    def _project(self, event: dict) -> None:
        turn = self._turn
        if turn is None:
            return
        kind = event.get("type")

        if kind == "system" and event.get("subtype") == "init":
            # The session filter is LEARNED here, never fixed by the client.
            # `--resume` emits the ORIGINAL id, so a client-fixed filter would
            # discard the entire stream of a resumed session.
            if self._session_id is None:
                self._session_id = event.get("session_id")
                turn.session_id = self._session_id
            return

        event_session = event.get("session_id")
        if event_session and self._session_id and event_session != self._session_id:
            turn.blocking_anomalies.append(f"event from foreign session {event_session!r}")
            return

        if kind == "assistant":
            message = event.get("message") or {}
            turn.messages.append(message)
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    turn.final_text = (turn.final_text or "") + (block.get("text") or "")
                elif block.get("type") == "tool_use":
                    turn.tool_calls.append(
                        ToolCall(
                            tool_use_id=block.get("id") or "",
                            tool_name=block.get("name") or "",
                            gated=False,
                        )
                    )
            usage = message.get("usage")
            if isinstance(usage, dict):
                turn.usage.update(usage)
            return

        if kind == "user":
            turn.messages.append(event.get("message") or {})
            return

        if kind == "result":
            self._project_result(event, turn)

    def _project_result(self, event: dict, turn: ClaudeTurnResult) -> None:
        turn.subtype = event.get("subtype")
        turn.is_error = bool(event.get("is_error"))
        turn.terminal_reason = event.get("terminal_reason")
        turn.num_turns = event.get("num_turns")
        cost = event.get("total_cost_usd")
        turn.total_cost_usd = float(cost) if isinstance(cost, (int, float)) else None
        if isinstance(event.get("usage"), dict):
            turn.usage.update(event["usage"])
        if turn.final_text is None and isinstance(event.get("result"), str):
            turn.final_text = event["result"]

        # Order matters, and the obvious order is wrong. An earlier revision
        # checked terminal_reason FIRST, so an API failure that ends with
        # terminal_reason="aborted_streaming" was reported as a user
        # cancellation — the only unhappy path it exercised, it misclassified.
        if turn.is_error or turn.subtype not in (None, "success"):
            turn.status = "error"
        elif turn.terminal_reason == "aborted_streaming" and turn.interrupted:
            turn.status = "cancelled"
        else:
            turn.status = "ok"

        if turn.projection_failures:
            # A turn that failed to project cannot be reported as ok: the
            # transcript is incomplete, typically a tool_use with no matching
            # tool_result, and a downstream supervisor would believe it.
            turn.status = "projection_error"

    # --- turns -------------------------------------------------------------

    def run_turn(self, text: str, *, timeout: float = 900.0) -> ClaudeTurnResult:
        if self._proc is None:
            raise ClaudeCodeSessionError("session not started")
        with self._state_lock:
            self._turn = ClaudeTurnResult(session_id=self._session_id)
        self._result_ev.clear()
        sent = self.send(
            {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": text}]},
            }
        )
        if not sent:
            raise ClaudeCodeSessionError("could not write the user message to the CLI")

        if not self._result_ev.wait(timeout=timeout):
            self.interrupt()
            with self._state_lock:
                turn = self._turn
                if turn is not None:
                    turn.interrupted = True
                    turn.status = "timeout"
                    turn.blocking_anomalies.append(f"no result within {timeout}s")
            self._result_ev.wait(timeout=15.0)

        with self._state_lock:
            turn = self._turn or ClaudeTurnResult()
            if self._reader_died:
                turn.status = "transport_error"
                turn.blocking_anomalies.append(f"reader thread died: {self._reader_died}")
            if turn.status == "unknown":
                turn.status = "no_result"
            self._turn = None
            return turn


def check_claude_binary(claude_bin: str = "claude") -> tuple[bool, str]:
    """Version preflight. Returns (ok, detail); never raises."""
    try:
        proc = subprocess.run(  # noqa: S603
            [claude_bin, "--version"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{claude_bin} is not runnable: {exc}"
    parsed = parse_claude_version(proc.stdout or proc.stderr)
    if parsed is None:
        return False, f"could not parse a version from {(proc.stdout or '').strip()!r}"
    if parsed < MIN_CLAUDE_VERSION:
        return False, f"claude {parsed} is older than the measured minimum {MIN_CLAUDE_VERSION}"
    return True, ".".join(str(p) for p in parsed)
