"""Regression harness for the dashboard auth gate.

Phase 0 — establish a baseline pin on the current (pre-OAuth) behavior so
later phases can prove they didn't break loopback mode.
"""
import pytest

# Phase 5 / Phase 6: these tests mutate ``web_server.app.state.auth_required``
# at module level. Run them in the same xdist worker so they don't race
# against each other (and against any other file that also touches
# ``app.state``) — the marker name is shared across all dashboard-auth test
# files that gate the app.
from fastapi.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client_loopback():
    # Pin the bound-host state for host_header_middleware so requests with
    # default Host: testclient pass the DNS-rebinding check.  TestClient
    # sends Host: testserver by default, but our middleware accepts the
    # loopback aliases when bound_host is loopback.
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port






# ---------------------------------------------------------------------------
# should_require_auth predicate (Task 0.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host,allow_public,expected", [
    ("127.0.0.1", False, False),
    ("127.0.0.1", True,  False),
    ("localhost", False, False),
    ("::1",       False, False),
    # --insecure (allow_public=True) NO LONGER bypasses the gate on a public
    # bind (June 2026 hermes-0day hardening). Non-loopback always requires auth.
    ("0.0.0.0",   True,  True),
    ("0.0.0.0",   False, True),
    ("192.168.1.5", False, True),
    ("10.0.0.1",  True,  True),     # allow_public ignored — LAN IP is public
    ("100.64.0.1", False, True),    # Tailscale CGNAT — treated as public
    ("hermes-agent-prod-abc.fly.dev", False, True),
])
def test_should_require_auth_truth_table(host, allow_public, expected):
    from hermes_cli.web_server import should_require_auth
    assert should_require_auth(host, allow_public) is expected


# ---------------------------------------------------------------------------
# start_server stashes auth_required on app.state (Task 0.3)
# ---------------------------------------------------------------------------


def _stub_uvicorn_run(monkeypatch):
    """Replace uvicorn.Config/Server with no-op fakes so start_server
    returns immediately (rather than blocking on the event loop). Returns the dict
    that will capture the keyword args.
    """
    import asyncio
    import contextlib
    import uvicorn
    captured: dict = {"kwargs": {}}

    class _FakeConfig:
        loaded = True
        host = "127.0.0.1"
        port = 8000

        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def load(self):
            pass

        class lifespan_class:
            should_exit = False
            state: dict = {}

            def __init__(self, *a, **kw):
                pass

            async def startup(self):
                pass

            async def shutdown(self):
                pass

    class _FakeServer:
        should_exit = False
        started = True
        servers: list = []
        lifespan = None

        @staticmethod
        def capture_signals():
            return contextlib.nullcontext()

        async def startup(self, sockets=None):
            pass

        async def main_loop(self):
            pass

        async def shutdown(self, sockets=None):
            pass

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", lambda config: _FakeServer())
    return captured


def test_start_server_loopback_sets_auth_required_false(monkeypatch):
    """Loopback bind: app.state.auth_required is False after start_server."""
    _stub_uvicorn_run(monkeypatch)
    # Force a fresh state to detect that start_server actually set it.
    web_server.app.state.auth_required = None
    web_server.start_server(
        host="127.0.0.1", port=9119,
        open_browser=False, allow_public=False,
    )
    assert web_server.app.state.auth_required is False


def test_start_server_insecure_public_no_longer_bypasses_gate(monkeypatch):
    """``--insecure`` (allow_public=True) on a public host: gate now ENGAGES.

    June 2026 hardening: --insecure no longer disables auth. With no providers
    registered, the bind fails closed (SystemExit) and auth_required is True.
    """
    from hermes_cli.dashboard_auth import clear_providers
    clear_providers()
    _stub_uvicorn_run(monkeypatch)
    web_server.app.state.auth_required = None
    with pytest.raises(SystemExit):
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=True,
        )
    assert web_server.app.state.auth_required is True


def test_start_server_public_without_insecure_records_auth_required(monkeypatch):
    """Public bind without --insecure: the gate engages and auth_required=True.

    With no providers registered, this fails closed with SystemExit. The
    flag-stashing happens BEFORE the exit so the rest of the system can
    branch on it. (See task 3.5 tests below for the with-provider path.)
    """
    from hermes_cli.dashboard_auth import clear_providers
    clear_providers()
    _stub_uvicorn_run(monkeypatch)
    web_server.app.state.auth_required = None
    with pytest.raises(SystemExit):
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=False,
        )
    assert web_server.app.state.auth_required is True


# ---------------------------------------------------------------------------
# Task 3.5: start_server fail-closed + proxy_headers + index-token suppression
# ---------------------------------------------------------------------------


def test_start_server_gate_with_provider_proceeds_and_sets_proxy_headers(monkeypatch):
    """With at least one provider, public bind + no --insecure starts the server.

    The SystemExit-refusing-to-bind guard is REPLACED in gated mode by
    "the gate engages", so as long as a provider is registered the bind
    succeeds.  uvicorn is called with proxy_headers=True so X-Forwarded-Proto
    from Fly's TLS terminator is honoured for cookie Secure-flag decisions.
    """
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    clear_providers()
    register_provider(StubAuthProvider())
    captured = _stub_uvicorn_run(monkeypatch)
    try:
        web_server.app.state.auth_required = None
        web_server.start_server(
            host="0.0.0.0", port=9119,
            open_browser=False, allow_public=False,
        )
        assert web_server.app.state.auth_required is True
        assert captured["kwargs"].get("host") == "0.0.0.0"
        assert captured["kwargs"].get("proxy_headers") is True
    finally:
        clear_providers()




# ---------------------------------------------------------------------------
# HERMES_DASHBOARD_FORCE_AUTH — the tunnel topology the bind cannot see
# ---------------------------------------------------------------------------
# A tunnel (cloudflared, ngrok) terminates a public hostname and forwards to
# 127.0.0.1, so the bind says "loopback" while the dashboard is on the
# internet. In loopback mode the SPA HTML carries the session token, so that
# misread publishes the credential. These tests pin the override's contract.


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " on "])
def test_force_auth_engages_the_gate_on_a_loopback_bind(monkeypatch, raw):
    from hermes_cli.web_server import should_require_auth

    monkeypatch.setenv("HERMES_DASHBOARD_FORCE_AUTH", raw)
    for host in ("127.0.0.1", "localhost", "::1"):
        assert should_require_auth(host) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "  ", "maybe"])
def test_force_auth_falsy_leaves_the_bind_heuristic_alone(monkeypatch, raw):
    from hermes_cli.web_server import should_require_auth

    monkeypatch.setenv("HERMES_DASHBOARD_FORCE_AUTH", raw)
    assert should_require_auth("127.0.0.1") is False
    assert should_require_auth("0.0.0.0") is True


def test_force_auth_can_never_disable_the_gate(monkeypatch):
    """The override is one-directional: no value of it turns auth OFF.

    This is the property that makes the flag safe to expose. A typo can cost
    us a login page; it must never cost us the gate on a public bind.
    """
    from hermes_cli.web_server import should_require_auth

    for raw in ("0", "false", "off", "no", "", "1", "true", "on", "garbage"):
        monkeypatch.setenv("HERMES_DASHBOARD_FORCE_AUTH", raw)
        assert should_require_auth("0.0.0.0") is True
        assert should_require_auth("192.168.1.5", allow_public=True) is True


def test_force_auth_absent_is_the_documented_default(monkeypatch):
    from hermes_cli.web_server import should_require_auth

    monkeypatch.delenv("HERMES_DASHBOARD_FORCE_AUTH", raising=False)
    assert should_require_auth("127.0.0.1") is False
    assert should_require_auth("0.0.0.0") is True


# ---------------------------------------------------------------------------
# HERMES_DASHBOARD_TRUSTED_PROXY_HOSTS — o alias que o Origin do WS exige
# ---------------------------------------------------------------------------
# O fork tinha o env var documentado e NAO lido: `_is_accepted_host` so aceitava
# nomes de loopback. Com um tunel na frente, o `Host` da para reescrever, mas o
# `Origin` do WebSocket e' do navegador -- o chat fechava com 1006 em laco.


def test_trusted_proxy_host_is_accepted_on_a_loopback_bind(monkeypatch):
    from hermes_cli.web_server import _is_accepted_host

    monkeypatch.setenv("HERMES_DASHBOARD_TRUSTED_PROXY_HOSTS", "hermes.example.com")
    assert _is_accepted_host("hermes.example.com", "127.0.0.1") is True
    assert _is_accepted_host("hermes.example.com:443", "127.0.0.1") is True
    assert _is_accepted_host("127.0.0.1:9119", "127.0.0.1") is True


def test_a_host_outside_the_allowlist_is_still_refused(monkeypatch):
    """The allowlist widens by exactly what the operator names, nothing more."""
    from hermes_cli.web_server import _is_accepted_host

    monkeypatch.setenv("HERMES_DASHBOARD_TRUSTED_PROXY_HOSTS", "hermes.example.com")
    for hostile in ("evil.example.com", "hermes.example.com.evil.net", "attacker"):
        assert _is_accepted_host(hostile, "127.0.0.1") is False


@pytest.mark.parametrize("wildcard", ["*", "0.0.0.0", "::"])
def test_a_wildcard_entry_never_disables_the_guard(monkeypatch, wildcard):
    from hermes_cli.web_server import _is_accepted_host

    monkeypatch.setenv("HERMES_DASHBOARD_TRUSTED_PROXY_HOSTS", wildcard)
    assert _is_accepted_host("evil.example.com", "127.0.0.1") is False


def test_absent_allowlist_keeps_the_previous_behaviour(monkeypatch):
    from hermes_cli.web_server import _is_accepted_host

    monkeypatch.delenv("HERMES_DASHBOARD_TRUSTED_PROXY_HOSTS", raising=False)
    assert _is_accepted_host("hermes.example.com", "127.0.0.1") is False
    assert _is_accepted_host("localhost:9119", "127.0.0.1") is True
