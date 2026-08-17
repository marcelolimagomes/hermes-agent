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


def _approval_prompt(tool_name: str, tool_input: Any) -> str:
    """Human-readable request for Hermes' approval prompt."""
    detail = ""
    if isinstance(tool_input, dict):
        detail = str(tool_input.get("command") or tool_input.get("file_path") or "")
    elif tool_input:
        detail = str(tool_input)
    detail = detail.strip()
    if len(detail) > 200:
        detail = detail[:200] + "…"
    return f"Claude Code lane wants to run {tool_name}" + (f": {detail}" if detail else "")


def make_claude_approval_bridge(agent) -> Callable[[Optional[str], Any, str], Any]:
    """Build the policy callable the transport consults before each tool.

    Resolution order, from most to least authoritative:

      1. Hermes' approval bypass (`approvals.mode: off`, /yolo, --yolo).
      2. Hermes' per-thread approval callback, the same one codex_runtime uses.
      3. Deny — gateway and cron have no UI to prompt through.

    Never "allow by default": on this runtime the gate is the whole boundary.
    """
    from agent.transports.claude_code_session import GateDecision

    def bridge(tool_name: Optional[str], tool_input: Any, channel: str) -> GateDecision:
        name = (tool_name or "").strip()
        if not name:
            return GateDecision(allow=False, reason="tool without a name")

        # 1. Explicit approval bypass, exactly as the Codex runtime honours it:
        #    `approvals.mode: off`, the /yolo session toggle, --yolo or
        #    HERMES_YOLO_MODE. When the operator has opted out of Hermes
        #    approvals, double-gating here would make the lane unusable while
        #    adding nothing — the same reasoning codex_runtime documents.
        try:
            from tools.approval import is_approval_bypass_active

            if is_approval_bypass_active():
                return GateDecision(
                    allow=True, reason=f"hermes approval bypass active ({channel})"
                )
        except Exception:  # noqa: BLE001
            logger.debug("claude_code_cli: approval-bypass lookup failed", exc_info=True)

        # 2. Hermes' standard per-thread approval callback, installed by the CLI
        #    thread. This is the SAME mechanism the Codex runtime uses; an
        #    earlier version of this bridge invented `request_tool_approval`,
        #    which exists nowhere, so every tool was denied and the lane looked
        #    safe while being unable to do any work.
        try:
            from tools.terminal_tool import _get_approval_callback

            approval_callback = _get_approval_callback()
        except Exception:  # noqa: BLE001
            approval_callback = None

        if callable(approval_callback):
            try:
                verdict = approval_callback(_approval_prompt(name, tool_input))
            except Exception as exc:  # noqa: BLE001
                logger.warning("claude_code_cli: approval callback raised: %s", exc)
                return GateDecision(
                    allow=False, reason=f"approval callback raised: {type(exc).__name__}"
                )
            allowed = verdict is True or str(verdict).strip().lower() in {"y", "yes", "approve", "allow"}
            return GateDecision(
                allow=allowed, reason=f"hermes approval callback via {channel}"
            )

        # 3. No UI and no bypass: gateway and cron contexts have nowhere to
        #    surface the prompt. Fail closed, which is what the Codex runtime
        #    does in the same situation.
        return GateDecision(
            allow=False,
            reason="no Hermes approval UI and no bypass: failing closed as codex_app_server does",
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
