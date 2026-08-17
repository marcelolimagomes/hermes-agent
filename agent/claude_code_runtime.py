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


def _resolve_mcp_env(servers: dict) -> dict:
    """Resolve the ${VAR} references the MCP specs need, for the SPAWN ENV.

    The Paperclip stdio server authenticates through environment variables that
    the profile stores as ${VAR} references. Hermes resolves them for its own
    MCP client; a CLI-driven lane spawns the server itself, so the values have
    to reach it another way.

    They go into the child process environment, never into the mcp-config file:
    a file on disk would be a materialised secret, an env var of a child we
    spawn is the same channel Hermes already uses.
    """
    import os
    import re

    wanted: set[str] = set()
    for spec in servers.values():
        for value in ((spec or {}).get("env") or {}).values():
            if isinstance(value, str):
                wanted.update(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value))
        for value in ((spec or {}).get("headers") or {}).values():
            if isinstance(value, str):
                wanted.update(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value))
    if not wanted:
        return {}

    resolved: dict[str, str] = {}
    for name in wanted:
        current = os.environ.get(name)
        if current:
            resolved[name] = current

    missing = wanted - set(resolved)
    if missing:
        # The profile .env is where Hermes keeps these, mode 600. Read only the
        # names the MCP specs actually asked for: never sweep the whole file.
        home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        profile = (os.environ.get("HERMES_PROFILE") or "").strip()
        if not profile:
            try:
                with open(os.path.join(home, "active_profile"), encoding="utf-8") as handle:
                    profile = handle.read().strip()
            except OSError:
                profile = ""
        # The default profile keeps its .env at HERMES_HOME; named profiles under
        # profiles/<name>/. Same layout iac/ai-memory-mcp.sh relies on.
        env_path = (
            os.path.join(home, "profiles", profile, ".env")
            if profile and profile != "default"
            else os.path.join(home, ".env")
        )
        if env_path and os.path.isfile(env_path):
            try:
                with open(env_path, encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip().removeprefix("export ").strip()
                        if key in missing:
                            resolved[key] = value.strip().strip('"').strip("'")
            except OSError:
                logger.debug("claude_code_cli: could not read the profile .env", exc_info=True)

    still_missing = sorted(wanted - set(resolved))
    if still_missing:
        # Name them, never their values: a server that will fail to connect is
        # worth saying out loud instead of surfacing as "tool not available".
        logger.warning(
            "claude_code_cli: MCP env unresolved, those servers will not connect: %s",
            ", ".join(still_missing),
        )
    return resolved


def _profile_mcp_servers(agent) -> dict:
    """Read the MCP servers from the ACTIVE PROFILE config.

    An earlier version read `agent.mcp_servers`, an attribute that does not
    exist: the lane silently got no servers, the model was told about no tools,
    and a turn asking for Paperclip answered "I could try an equivalent probe
    via Bash". Hermes keeps them in the profile config — same source
    `agent/coding_context.py:705` reads — and only enabled ones count.
    """
    override = getattr(agent, "mcp_servers", None)
    if isinstance(override, dict) and override:
        return override
    try:
        from hermes_cli.config import read_raw_config
        from hermes_cli.tools_config import _parse_enabled_flag

        servers = read_raw_config().get("mcp_servers") or {}
    except Exception:  # noqa: BLE001
        logger.debug("claude_code_cli: could not read profile mcp_servers", exc_info=True)
        return {}
    if not isinstance(servers, dict):
        return {}
    return {
        name: spec
        for name, spec in servers.items()
        if isinstance(spec, dict)
        and _parse_enabled_flag(spec.get("enabled", True), default=True)
    }


def _mcp_config_from_profile(agent) -> Optional[str]:
    """Translate Hermes' configured MCP servers into a Claude CLI mcp-config.

    Without this the lane has NO Paperclip and NO ai-memory tools: the
    orchestrator could not read its issue, post results, or reach memory. The
    Codex runtime gets these through Hermes' own tool dispatch; this runtime
    hands the whole turn to the CLI, so the servers must be handed over too.

    Only the allowlisted tools are carried across. The lane allowlist is a
    contract (config/contracts/mcp-lanes.toml in the appliance), and widening
    it here would move a security boundary as a side effect of a translation.
    """
    import json
    import os
    import tempfile

    servers = _profile_mcp_servers(agent)
    if not servers:
        return None

    translated: dict[str, Any] = {}
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("command"):
            entry: dict[str, Any] = {
                "command": spec["command"],
                "args": list(spec.get("args") or []),
            }
            # Env is deliberately NOT copied when it holds ${VAR} references.
            # The CLI does not expand them, so the server would receive the
            # literal string and fail to authenticate. Expanding them here
            # would write real credentials into a temp file, which this product
            # forbids. The MCP server is spawned BY the CLI, which inherits our
            # process environment, so resolved values reach it by inheritance —
            # no secret is ever serialised.
            literal_env = {
                key: value
                for key, value in (spec.get("env") or {}).items()
                if isinstance(value, str) and "${" not in value
            }
            if literal_env:
                entry["env"] = literal_env
        elif spec.get("url"):
            # Claude Code speaks http/sse MCP; Hermes stores the URL plus any
            # headers, which is where the ai-memory bearer lives. The value is
            # an ${ENV} reference in the profile and stays one here: no
            # credential is ever materialised into this file.
            entry = {"type": "http", "url": spec["url"]}
            if isinstance(spec.get("headers"), dict):
                entry["headers"] = dict(spec["headers"])
        else:
            continue
        translated[name] = entry

    if not translated:
        return None

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".mcp.json", prefix="hermes-claude-", delete=False
    )
    with handle:
        json.dump({"mcpServers": translated}, handle)
    os.chmod(handle.name, 0o600)
    return handle.name


def _allowed_mcp_tools(agent) -> list[str]:
    """The exact tool names the lane may call, in Claude's mcp__ namespace."""
    names: list[str] = []
    servers = _profile_mcp_servers(agent)
    for server, spec in servers.items():
        include = ((spec or {}).get("tools") or {}).get("include") or []
        for tool in include:
            names.append(f"mcp__{server}__{tool}")
    return names


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
        mcp_servers = _profile_mcp_servers(agent)
        mcp_config = _mcp_config_from_profile(agent)
        mcp_tools = _allowed_mcp_tools(agent)
        mcp_env = _resolve_mcp_env(mcp_servers)
        # The tool surface must carry the MCP tools explicitly: an empty or
        # Bash-only surface makes the orchestrator unable to touch Paperclip.
        tools = ",".join(["Bash", "Read", "Write", "Edit", *mcp_tools]) or "Bash"
        session = ClaudeCodeSession(
            claude_bin=getattr(agent, "claude_bin", "claude") or "claude",
            identity=Identity.new(),
            cwd=getattr(agent, "workspace", None) or None,
            tools=tools,
            model=getattr(agent, "model", None),
            mcp_config=mcp_config,
            env=mcp_env or None,
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
