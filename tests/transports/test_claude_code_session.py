"""Behavioural contract for the Claude Code stream-json transport.

Every assertion here encodes a measurement against `claude` 2.1.204, or a
defect that a previous implementation actually shipped. Tests that only pass
prove nothing: the projection cases feed the SAME event that broke the earlier
driver.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.transports import claude_code_session as ccs

# This suite runs WITHOUT pytest: the appliance gates invoke it directly and
# adding a test dependency to the gate path would be a new dependency to pin.
# It still works under pytest, where the plain asserts and the local `raises`
# helper behave identically.


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} and nothing was raised")


def _session() -> ccs.ClaudeCodeSession:
    s = ccs.ClaudeCodeSession(identity=ccs.Identity.new(), register_hook=False)
    s._turn = ccs.ClaudeTurnResult()
    return s


def test_version_is_parsed_not_compared_literally():
    # `claude --version` prints "2.1.204 (Claude Code)": the suffix trails the
    # semver, so prefix equality against a composed string always fails.
    assert ccs.parse_claude_version("2.1.204 (Claude Code)") == (2, 1, 204)
    assert ccs.parse_claude_version("no version here") is None


def test_argv_refuses_barrier_removal():
    for bad in (["--dangerously-skip-permissions"],
                ["--allow-dangerously-skip-permissions"],
                ["--permission-mode", "bypassPermissions"]):
        with raises(ccs.ClaudeCodeIdentityError):
            ccs.build_argv("claude", ccs.Identity.new(), extra_args=bad)


def test_empty_tool_surface_is_refused():
    # Measured: --tools "" yields system/init with tools=[], which makes the
    # approval gate unreachable. An untestable gate is not a gate.
    with raises(ccs.ClaudeCodeIdentityError):
        ccs.build_argv("claude", ccs.Identity.new(), tools="")


def test_resume_and_session_id_are_mutually_exclusive():
    # Measured: the CLI exits 1 with an explicit message. Refuse before spawn.
    with raises(ccs.ClaudeCodeIdentityError):
        ccs.Identity(form="resume", parent_session_id="a", session_id="b").validate()
    assert ccs.Identity.fork("a", "b").argv_fragment()[-1] == "--fork-session"


def test_tool_use_result_may_be_str_dict_or_list():
    # The CLI delivers tool_use_result as a STRING for at least Bash errors.
    # The earlier projector assumed dict and died with AttributeError.
    session = _session()
    for payload in ("Error: exit 1", {"stdout": "ok"}, [{"type": "text"}]):
        event = {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": payload}]}}
        session._project(event)
    assert session._turn.projection_failures == []


def test_projection_failure_forbids_an_ok_turn():
    # A turn whose projection blew up must not be reported ok: the transcript
    # is incomplete and a downstream supervisor would believe it.
    session = _session()
    session._turn.projection_failures.append({"line": 2, "error": "boom"})
    session._project_result({"type": "result", "subtype": "success", "is_error": False}, session._turn)
    assert session._turn.status == "projection_error"


def test_api_failure_is_not_reported_as_cancellation():
    # Order of evaluation: is_error/subtype decide, terminal_reason qualifies.
    # The earlier driver checked terminal_reason first and sold an API failure
    # as a user cancellation — the only unhappy path it ran, it got wrong.
    session = _session()
    session._project_result(
        {"type": "result", "subtype": "error_during_execution", "is_error": True,
         "terminal_reason": "aborted_streaming"},
        session._turn)
    assert session._turn.status == "error"


def test_cancellation_requires_our_own_interrupt():
    session = _session()
    session._turn.interrupted = True
    session._project_result(
        {"type": "result", "subtype": "success", "is_error": False,
         "terminal_reason": "aborted_streaming"},
        session._turn)
    assert session._turn.status == "cancelled"


def test_session_filter_is_learned_from_init_not_fixed():
    # --resume emits the ORIGINAL id. A client-fixed filter would discard the
    # whole stream of a resumed session.
    session = _session()
    session._project({"type": "system", "subtype": "init", "session_id": "S-real"})
    assert session._session_id == "S-real"
    session._project({"type": "assistant", "session_id": "S-real",
                      "message": {"content": [{"type": "text", "text": "hi"}]}})
    assert session._turn.final_text == "hi"
    session._project({"type": "assistant", "session_id": "S-other",
                      "message": {"content": [{"type": "text", "text": "leak"}]}})
    assert session._turn.blocking_anomalies, "foreign-session event must be an anomaly"
    assert "leak" not in (session._turn.final_text or "")


def test_malformed_control_request_does_not_kill_the_reader():
    # The exact event that killed an earlier revision: `request` as a string.
    # Control dispatch and projection must share one guard.
    session = _session()
    with raises(ccs.ClaudeCodeSessionError):
        session._on_control_request({"type": "control_request", "request": "not-an-object"})


def test_ungated_tool_is_recorded_as_a_bypass():
    # A transport that records only denials cannot distinguish "allowed" from
    # "never asked". gate_bypasses is what makes the audit possible.
    session = _session()
    session._project({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t9", "name": "Bash"}]}})
    assert [c.tool_use_id for c in session._turn.gate_bypasses] == ["t9"]
    session._note_gate("t9", "Bash", ccs.GateDecision(allow=False, reason="policy"))
    assert session._turn.gate_bypasses == []
    assert session._turn.permission_denials


def test_policy_that_raises_becomes_a_denial():
    def exploding(_n, _i, _c):
        raise RuntimeError("policy bug")

    session = ccs.ClaudeCodeSession(identity=ccs.Identity.new(), policy=exploding, register_hook=False)
    assert session._safe_policy("Bash", {}, "hook_callback").allow is False


def test_default_policy_is_deny():
    assert ccs.default_policy("Bash", {"command": "ls"}, "hook_callback").allow is False


def test_send_serializes_under_concurrency():
    # Not a docstring check: two interleaved lines do not lose an event, they
    # KILL the CLI process. Chunked writes with jitter are the real window.
    import io, random, time

    class Chunky:
        # Not io.TextIOBase: its `closed` is a read-only property, and send()
        # checks that attribute before writing.
        def __init__(self):
            self.buf = []
            self.closed = False
        def write(self, data):
            for i in range(0, len(data), 7):
                self.buf.append(data[i:i + 7])
                time.sleep(random.uniform(0, 0.0003))
            return len(data)
        def flush(self):
            return None

    session = ccs.ClaudeCodeSession(identity=ccs.Identity.new(), register_hook=False)
    fake = Chunky()
    session._proc = type("P", (), {"stdin": fake, "poll": lambda self: None})()

    def worker(tid):
        for n in range(12):
            session.send({"type": "user", "tid": tid, "n": n, "pad": "x" * 40})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = "".join(fake.buf).splitlines()
    assert len(lines) == 8 * 12
    for line in lines:
        json.loads(line)  # raises if two writes interleaved


def test_unserializable_payload_never_reaches_stdin():
    session = ccs.ClaudeCodeSession(identity=ccs.Identity.new(), register_hook=False)
    session._proc = type("P", (), {"stdin": type("S", (), {"closed": False})(), "poll": lambda self: None})()
    with raises(ccs.ClaudeCodeSessionError):
        session.send({"bad": {1, 2, 3}})


def test_provider_is_registered_as_external_process():
    # ADR-020 authorizes registration of the CLI transport, and only that.
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY["claude-code-cli"]
    assert cfg.auth_type == "external_process"
    # The base URL is a marker, not an endpoint: an http(s) URL here would mean
    # the turn goes to the Anthropic API, which is exactly what is forbidden.
    assert cfg.inference_base_url.startswith("cli://")


def test_api_mode_is_registered():
    from hermes_cli.runtime_provider import _VALID_API_MODES

    assert "claude_code_cli" in _VALID_API_MODES


def test_registered_is_not_promoted():
    # The provider must NOT be auto-injected into the model picker. Hermes
    # already skips auto-injection for external_process flows; this asserts the
    # property the ADR depends on, so a future change that starts surfacing it
    # fails here instead of silently promoting the lane.
    import hermes_cli.models as models

    canonical = getattr(models, "CANONICAL_PROVIDERS", {}) or {}
    assert "claude-code-cli" not in canonical


# --- ponte de aprovação e despacho do runtime ---------------------------------

def test_approval_bridge_denies_without_any_mechanism():
    # Sem hook e sem allowlist, a lane fica sem supervisor. Permitir aqui
    # tornaria o turno irrestrito num runtime que nao tem sandbox.
    from agent.claude_code_runtime import make_claude_approval_bridge

    bridge = make_claude_approval_bridge(type("A", (), {})())
    assert bridge("Bash", {"command": "ls"}, "hook_callback").allow is False


def test_approval_bridge_uses_the_hermes_hook():
    from agent.claude_code_runtime import make_claude_approval_bridge

    class Agent:
        def request_tool_approval(self, name, _input):
            return name == "Bash"

    bridge = make_claude_approval_bridge(Agent())
    assert bridge("Bash", {}, "hook_callback").allow is True
    assert bridge("Write", {}, "hook_callback").allow is False


def test_approval_hook_that_raises_becomes_a_denial():
    from agent.claude_code_runtime import make_claude_approval_bridge

    class Agent:
        def request_tool_approval(self, name, _input):
            raise RuntimeError("hook bug")

    assert make_claude_approval_bridge(Agent())("Bash", {}, "hook_callback").allow is False


def test_non_boolean_verdict_is_ambiguous_and_denies():
    from agent.claude_code_runtime import make_claude_approval_bridge

    class Agent:
        def request_tool_approval(self, name, _input):
            return "sure"

    assert make_claude_approval_bridge(Agent())("Bash", {}, "hook_callback").allow is False


def test_allowlist_is_honoured_when_no_hook_exists():
    from agent.claude_code_runtime import make_claude_approval_bridge

    agent = type("A", (), {"allowed_tools": {"Bash"}})()
    bridge = make_claude_approval_bridge(agent)
    assert bridge("Bash", {}, "can_use_tool").allow is True
    assert bridge("Write", {}, "can_use_tool").allow is False


def test_api_mode_is_accepted_by_agent_init():
    # Sem isto o turno nao e despachado: o provider fica registrado e inerte.
    import inspect

    from agent import agent_init

    src = inspect.getsource(agent_init)
    assert '"claude_code_cli"' in src, "agent_init nao aceita o api_mode"


def test_conversation_loop_dispatches_the_runtime():
    import inspect

    from agent import conversation_loop

    src = inspect.getsource(conversation_loop)
    assert 'agent.api_mode == "claude_code_cli"' in src
    assert "run_claude_code_turn" in src


def test_provider_never_falls_through_to_the_native_api():
    # claude-code-cli tem de derivar claude_code_cli, JAMAIS anthropic_messages,
    # que e o caminho da API nativa proibido pelo ADR-020.
    import inspect

    from agent import agent_init

    src = inspect.getsource(agent_init)
    idx = src.index('agent.provider == "claude-code-cli"')
    branch = src[idx: idx + 400].split("elif")[0]
    # Comentarios sao descartados: o comentario deste proprio ramo explica por
    # que anthropic_messages e proibido, e olhar o texto cru reprovaria o
    # codigo correto por causa da sua propria justificativa.
    code = "\n".join(
        line for line in branch.splitlines() if not line.strip().startswith("#")
    )
    assert 'agent.api_mode = "claude_code_cli"' in code
    assert "anthropic_messages" not in code


def test_external_process_resolver_is_provider_aware():
    # Este resolver era hardcoded para o Copilot: um SEGUNDO provider
    # external_process resolvia o binario `copilot`, nao encontrava, e o
    # chamador caia em outro fornecedor EM SILENCIO. Medido: a lane ia parar
    # em openrouter.
    from hermes_cli.auth import resolve_external_process_provider_credentials

    creds = resolve_external_process_provider_credentials("claude-code-cli")
    assert creds["provider"] == "claude-code-cli"
    assert creds["command"].endswith("claude")
    assert creds["source"] == "process"


def test_provider_routing_never_yields_anthropic_messages():
    # O roteamento tem de devolver claude_code_cli. anthropic_messages levaria
    # o turno para a API nativa, proibida pelo ADR-020.
    import inspect

    from hermes_cli import runtime_provider

    src = inspect.getsource(runtime_provider)
    idx = src.index('if provider == "claude-code-cli":')
    branch = src[idx: idx + 700]
    code = "\n".join(l for l in branch.splitlines() if not l.strip().startswith("#"))
    assert '"api_mode": "claude_code_cli"' in code
    assert "anthropic_messages" not in code.split('if provider == "copilot-acp"')[0]


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK ] {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\nclaude code transport contract: {len(tests) - len(failed)}/{len(tests)} passaram")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
