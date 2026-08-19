"""O loop de LSP nao pode derrubar o turno quando o selector morre.

Sintoma medido em 2026-08-18: `Exception in thread hermes-lsp-loop:
OSError: [Errno 9] Bad file descriptor` no stderr de uma run do Paperclip. O
processo saiu com codigo 0, mas quem le o stderr reportou a run como falha.
"""

import threading
import time

import pytest

from agent.lsp.manager import _BackgroundLoop


class _LoopQueBrota:
    """Loop falso cujo run_forever() levanta EBADF, como o selector faz quando
    um descritor observado e' fechado por baixo dele. Exercita a guarda deste
    modulo sem depender de detalhes internos do asyncio."""

    def __init__(self):
        self.closed = False

    def run_forever(self):
        raise OSError(9, "Bad file descriptor")

    def close(self):
        self.closed = True


def test_uma_excecao_no_loop_nao_escapa_da_thread(monkeypatch, caplog):
    falso = _LoopQueBrota()
    monkeypatch.setattr("asyncio.new_event_loop", lambda: falso)
    monkeypatch.setattr("asyncio.set_event_loop", lambda _loop: None)

    loop = _BackgroundLoop()
    with caplog.at_level("WARNING", logger="agent.lsp.manager"):
        loop._run_forever()

    assert "hermes-lsp-loop" in caplog.text
    assert falso.closed, "o loop deve ser fechado mesmo quando run_forever levanta"
    assert loop._loop is None, "a referencia morta deve ser solta para start() poder recriar"


def test_o_loop_e_recriavel_depois_de_morrer(monkeypatch):
    falso = _LoopQueBrota()
    monkeypatch.setattr("asyncio.new_event_loop", lambda: falso)
    monkeypatch.setattr("asyncio.set_event_loop", lambda _loop: None)

    loop = _BackgroundLoop()
    loop.start()
    _wait_until(lambda: loop._thread is not None and not loop._thread.is_alive())
    assert loop._loop is None

    monkeypatch.undo()
    loop.start()
    assert loop._thread is not None and loop._thread.is_alive()
    assert loop._loop is not None
    loop.stop()


def test_stop_permite_um_start_seguinte():
    """stop() deixava `_thread` preenchido, e start() virava no-op para sempre."""
    loop = _BackgroundLoop()
    loop.start()
    loop.stop()
    assert loop._thread is None

    loop.start()
    assert loop._thread is not None and loop._thread.is_alive()
    loop.stop()


def test_run_sem_loop_falha_rapido_em_vez_de_bloquear():
    loop = _BackgroundLoop()
    with pytest.raises(RuntimeError):
        loop.run(_noop_coro())


async def _noop_coro():
    return None


def _raise_ebadf():
    raise OSError(9, "Bad file descriptor")


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condicao nao ocorreu dentro do timeout")
