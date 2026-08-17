"""Claude Code CLI runtime path.

Hands an entire turn to a local `claude` subprocess over stream-json and
projects its events back into Hermes' messages list, so memory and skill review
keep working. Mirrors `agent/codex_runtime.py`, which does the same for the
Codex CLI.

Called from run_conversation() when `agent.api_mode == "claude_code_cli"`.
Returns the same dict shape as the chat_completions path.

Two properties this module exists to guarantee, both learned from measurement:

1. The approval gate must be bridged to Hermes. The transport's own default
   policy denies everything — correct as a default, useless as a lane. Without
   the bridge below, every tool call would be refused and the lane would look
   "safe" while being unable to do any work.

2. This runtime is NOT sandboxed. The Codex path runs with
   `sandbox_mode="workspace-write"` and explicit `writable_roots`; the Claude
   CLI has no equivalent. Containment here is a policy gate over a live
   supervisor, so the approval bridge is the ONLY thing between the model and
   the host. That is why an unknown tool, a policy error and a missing
   approval hook all resolve to deny.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def make_claude_approval_bridge(agent) -> Callable[[Optional[str], Any, str], Any]:
    """Build the policy callable the transport consults before each tool.

    Resolution order, from most to least authoritative:

      1. An explicit Hermes approval hook, when the host exposes one.
      2. The agent's own allowlist of tools, when configured.
      3. Deny.

    Never "allow by default": on this runtime the gate is the whole boundary.
    """
    from agent.transports.claude_code_session import GateDecision

    def bridge(tool_name: Optional[str], tool_input: Any, channel: str) -> GateDecision:
        name = (tool_name or "").strip()
        if not name:
            return GateDecision(allow=False, reason="tool without a name")

        # 1. Host-provided approval hook. Signature is kept loose on purpose:
        #    Hermes exposes approval through more than one shape across
        #    versions, and a TypeError here must not become an allow.
        hook = getattr(agent, "request_tool_approval", None)
        if callable(hook):
            try:
                verdict = hook(name, tool_input)
            except Exception as exc:  # noqa: BLE001
                logger.warning("claude_code_cli: approval hook raised: %s", exc)
                return GateDecision(allow=False, reason=f"approval hook raised: {type(exc).__name__}")
            if isinstance(verdict, bool):
                return GateDecision(
                    allow=verdict,
                    reason=f"hermes approval hook via {channel}",
                )
            # A non-boolean verdict is ambiguous, and ambiguity resolves to deny.
            return GateDecision(allow=False, reason="approval hook returned a non-boolean verdict")

        # 2. Explicit allowlist on the agent, when the deployment configured one.
        allowed = getattr(agent, "allowed_tools", None)
        if isinstance(allowed, (set, list, tuple)) and allowed:
            if name in set(allowed):
                return GateDecision(allow=True, reason=f"tool in agent allowlist via {channel}")
            return GateDecision(allow=False, reason=f"tool {name!r} is not in the agent allowlist")

        # 3. No approval mechanism configured. Deny, and say so — a lane that
        #    silently allowed here would be unsupervised, not permissive.
        return GateDecision(
            allow=False,
            reason="no Hermes approval mechanism is configured for the claude_code_cli runtime",
        )

    return bridge


def _project_messages(turn, messages: List[Dict[str, Any]]) -> int:
    """Append the turn's projected messages, returning how many were added."""
    added = 0
    for message in turn.messages:
        if not isinstance(message, dict) or not message:
            continue
        messages.append(message)
        added += 1
    if turn.final_text and not any(
        m.get("role") == "assistant" and m.get("content") == turn.final_text for m in turn.messages
    ):
        messages.append({"role": "assistant", "content": turn.final_text})
        added += 1
    return added


def run_claude_code_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Drive one turn through the Claude Code CLI and project it back."""
    from agent.transports.claude_code_session import (
        ClaudeCodeSession,
        Identity,
        check_claude_binary,
    )

    ok, detail = check_claude_binary(getattr(agent, "claude_bin", "claude") or "claude")
    if not ok:
        # Fail loudly at the boundary instead of producing an empty turn that
        # the caller would read as "the model had nothing to say".
        return {
            "final_response": None,
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "interrupted": False,
            "error": f"claude CLI unusable: {detail}",
            "agent_persisted": False,
        }

    session = getattr(agent, "_claude_session", None)
    if session is None or not session.is_alive():
        # One session per AIAgent, reused across turns: the CLI keeps context
        # between turns in the same process, which is why this transport is
        # session-persistent rather than process-per-turn.
        session = ClaudeCodeSession(
            claude_bin=getattr(agent, "claude_bin", "claude") or "claude",
            identity=Identity.new(),
            cwd=getattr(agent, "workspace", None) or None,
            model=getattr(agent, "model", None),
            policy=make_claude_approval_bridge(agent),
            register_hook=True,
        )
        session.start()
        agent._claude_session = session

    try:
        turn = session.run_turn(
            user_message, timeout=float(getattr(agent, "claude_turn_timeout", 900.0))
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("claude_code_cli: turn failed")
        return {
            "final_response": None,
            "messages": messages,
            "api_calls": 1,
            "completed": False,
            "partial": True,
            "interrupted": False,
            "error": f"{type(exc).__name__}: {exc}",
            "agent_persisted": False,
        }

    _project_messages(turn, messages)

    # A bypass is a tool that ran WITHOUT the gate being consulted. It is not
    # an error of the turn, but it is a supervision gap and must not be
    # invisible: on a runtime with no sandbox, this is the number that matters.
    if turn.gate_bypasses:
        logger.warning(
            "claude_code_cli: %d tool call(s) ran without passing the approval gate: %s",
            len(turn.gate_bypasses),
            ", ".join(sorted({c.tool_name for c in turn.gate_bypasses})),
        )

    error = None
    if turn.status in {"error", "transport_error", "projection_error", "no_result", "timeout"}:
        error = turn.status
        if turn.blocking_anomalies:
            error = f"{turn.status}: {'; '.join(turn.blocking_anomalies)}"

    usage = dict(turn.usage or {})
    agent._last_turn_usage = usage or None

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": turn.status == "ok",
        "partial": turn.status != "ok",
        "interrupted": turn.interrupted,
        "error": error,
        "agent_persisted": False,
        "claude_session_id": turn.session_id,
        "claude_gate_bypasses": len(turn.gate_bypasses),
        "claude_permission_denials": len(turn.permission_denials),
        "usage": usage,
    }
