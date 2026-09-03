"""Tests for the authenticated ``kirocrew pod api`` front door."""

from __future__ import annotations

import argparse
import http.client
import inspect
import json
import os
import threading
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from kiro_crew import platform_compat
from kiro_crew.instances import run_marker
from kiro_crew.pod import cli as pod_cli
from kiro_crew.pod import runtime as rt
from kiro_crew.pod.config import PodConfig


class _RecordingServer(HTTPServer):
    seen: list[dict[str, str]]
    token: str


class _Handler(BaseHTTPRequestHandler):
    """Small real HTTP peer that enforces the dashboard's query-token contract."""

    def _respond(self, *, include_body: bool = True) -> None:
        server = cast(_RecordingServer, self.server)
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        length = int(self.headers.get("Content-Length") or 0)
        sent = self.rfile.read(length).decode("utf-8") if length else ""
        server.seen.append(
            {
                "method": self.command,
                "target": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "sent": sent,
            }
        )
        if query.get("token") != [server.token]:
            self.send_response(403)
            self.end_headers()
            return
        if parsed.path.endswith("/boom"):
            self.send_response(503)
            payload = {"error": "unavailable"}
        elif parsed.path.endswith("/echo-token"):
            self.send_response(200)
            payload = {"target": self.path, "token": query["token"][0]}
        elif parsed.path.endswith("/echo-escaped-token"):
            self.send_response(200)
            token = query["token"][0]
            escaped = "".join(f"\\u{ord(char):04x}" for char in token)
            raw = urllib.parse.quote(token, safe="")
            plus = urllib.parse.quote_plus(token, safe="")
            body = f'{{"{escaped}":{{"raw":"{token}","url":"{raw}","plus":"{plus}"}}}}'
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if include_body:
                self.wfile.write(body.encode("utf-8"))
            return
        else:
            self.send_response(200)
            payload = {"ok": True, "sent": sent}
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if include_body:
            self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:
        self._respond()

    def do_HEAD(self) -> None:
        self._respond(include_body=False)

    def do_POST(self) -> None:
        self._respond()

    def do_DELETE(self) -> None:
        self._respond()

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture
def stub_gateway(monkeypatch: pytest.MonkeyPatch) -> _RecordingServer:
    server = _RecordingServer(("127.0.0.1", 0), _Handler)
    server.seen = []
    server.token = "fixture-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(rt, "is_active", lambda cfg, name: True)
    monkeypatch.setattr(rt, "derive_port", lambda cfg, name: server.server_address[1])
    monkeypatch.setattr(rt, "mint_token", lambda cfg, name, ttl="2h": "fixture-token")
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


#: Stand-in start-time identity for a pid that does not exist on the test host.
#: Tests that pin ``main_pid`` to a synthetic pid must pin the probe too, or the
#: freshness check would decline for the uninteresting reason that nothing is
#: there to ask -- which would let an ownership case pass VACUOUSLY.
_START_TOKEN_FOR_4242 = "test-start-identity-4242"


def _live_start_token() -> str:
    """Real start-time identity of this test process, asserted to be readable."""
    token = run_marker.pid_start_token(os.getpid())
    assert token, "this host cannot read its own process start time"
    return token


def _pin_start_token(monkeypatch: pytest.MonkeyPatch, pid: int, token: str) -> None:
    """Make the start-time probe answer *token* for *pid* and nothing else."""
    real = run_marker.pid_start_token
    monkeypatch.setattr(
        run_marker,
        "pid_start_token",
        lambda value: token if int(value) == pid else real(value),
    )


def _pid_record_path(cfg: PodConfig, name: str, port: int) -> Path:
    """Path of pod *name*'s gateway pid sidecar, as the pod's gateway writes it."""
    return cfg.home_dir(name) / run_marker.RUN_DIR_NAME / run_marker.pid_file_name(port)


def _write_record(
    cfg: PodConfig, name: str, port: int, pid_body: str, start: str | None = None
) -> Path:
    """Plant pod *name*'s pid sidecar, plus its ``.start`` sidecar when given.

    The start identity is a SEPARATE file on purpose: the pid file must stay a
    bare pid so the whole-file ``isdigit()`` reader shipped in older clients keeps
    parsing it. The ``.start`` suffix is spelled literally here, so renaming it in
    the module without updating its readers turns these tests red.
    """
    path = _pid_record_path(cfg, name, port)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pid_body, encoding="utf-8")
    if start is not None:
        path.with_suffix(".start").write_text(start, encoding="utf-8")
    return path


class TestPodRecordFreshness:
    """The pid record must prove it still names the process it was written for.

    A sidecar outlives a crash (``clear_marker`` runs only on a graceful stop),
    so a bare pid can afterwards name an unrelated process that inherited the
    number. That single leg is the whole reason listener attribution was ever
    treated as a precondition. Binding the record to the process's start time
    closes it at the source, so pid/MainPID agreement can be sufficient again --
    which is what an agent driving ``pod api`` from a context that cannot
    attribute another process's socket depends on.
    """

    @pytest.fixture
    def cfg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PodConfig:
        monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
        monkeypatch.setattr(rt, "IS_POSIX", True)
        return PodConfig.load()

    def test_an_unattributable_loopback_listener_still_proves_our_pod(
        self, cfg: PodConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE REGRESSION, in the shape the host actually produces.

        ``lsof`` is installed and runs fine; it simply cannot see a socket held
        by a process outside the caller's view -- which is every pod gateway,
        because the service manager started it and ``pod api`` asks from an
        unprivileged agent process. The lookup returns ``[]``, and treating that
        as undecidable withholds the credential from a pod that is demonstrably
        serving, permanently.
        """
        live = os.getpid()
        _write_record(cfg, "demo", 7999, f"{live}\n", _live_start_token())
        monkeypatch.setattr(rt, "main_pid", lambda c, n: live)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
        monkeypatch.setattr(rt, "find_port_listeners", lambda port: [])
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_POD

    def test_a_failed_listener_lookup_does_not_revoke_a_fresh_record(
        self, cfg: PodConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "The lookup broke" is not evidence about who holds the port."""

        def _boom(port: int) -> list[platform_compat.PortListener]:
            raise OSError("lsof exploded")

        live = os.getpid()
        _write_record(cfg, "demo", 7999, f"{live}\n", _live_start_token())
        monkeypatch.setattr(rt, "main_pid", lambda c, n: live)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
        monkeypatch.setattr(rt, "find_port_listeners", _boom)
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_POD

    def test_a_recycled_pid_cannot_attest_and_the_secret_stays_home(
        self, cfg: PodConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The chain the fail-closed change was reaching for, closed at its root.

        The record names a live pid and the service manager agrees, so pid
        equality alone would attest. The start time does not match, which is the
        one fact a process that merely inherited the number cannot forge, so the
        credential is withheld with no listener evidence involved at all.
        """
        live = os.getpid()
        home = cfg.home_dir("demo")
        home.mkdir(parents=True, exist_ok=True)
        (home / ".local_secret").write_text("must-not-leave-this-process")
        _write_record(cfg, "demo", 7999, f"{live}\n", "some-earlier-generation")
        monkeypatch.setattr(rt, "main_pid", lambda c, n: live)
        monkeypatch.setattr(rt, "derive_port", lambda c, n: 7999)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: False)
        monkeypatch.setattr(
            rt,
            "loopback_urlopen",
            lambda *a, **k: pytest.fail("the secret must not be sent"),
        )
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_UNPROVEN
        with pytest.raises(rt.PodOwnershipUnproven):
            rt.mint_token(cfg, "demo", "1h")

    def test_a_host_that_will_not_report_a_start_time_fails_closed(
        self, cfg: PodConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows has no start-identity implementation and answers ``None``.

        "Cannot confirm" must refuse, never silently accept: an unreadable start
        time leaves the record indistinguishable from a crash leftover.
        """
        live = os.getpid()
        _write_record(cfg, "demo", 7999, f"{live}\n", "recorded-identity")
        monkeypatch.setattr(rt, "main_pid", lambda c, n: live)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: False)
        monkeypatch.setattr(platform_compat, "get_process_start_id", lambda pid: None)
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_UNPROVEN

    def test_a_fresh_record_still_loses_to_a_disagreeing_listener(
        self, cfg: PodConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive attribution outranks the record; corroboration cuts both ways."""
        live = os.getpid()
        _write_record(cfg, "demo", 7999, f"{live}\n", _live_start_token())
        monkeypatch.setattr(rt, "main_pid", lambda c, n: live)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
        monkeypatch.setattr(
            rt,
            "find_port_listeners",
            lambda port: [platform_compat.PortListener(live + 1, "127.0.0.1", "4")],
        )
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_FOREIGN

    def test_the_gateway_records_its_own_start_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Write side: a real ``write_marker`` produces a record that verifies.

        Without this the reader could be correct about a format nothing emits.
        The pid file must ALSO stay a bare pid, because the reader shipped on
        older clients takes the whole file and requires ``isdigit()``.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
        run_marker.write_marker(7999)
        pid_file = run_marker.pid_path(7999)
        assert pid_file.read_bytes() == f"{os.getpid()}\n".encode()
        assert pid_file.with_suffix(".start").read_text(encoding="utf-8").strip() == (
            _live_start_token()
        )
        record = run_marker.read_pid_record_path(pid_file)
        assert record is not None
        pid, started = record
        assert pid == os.getpid()
        assert started == _live_start_token()

    def test_removing_the_start_sidecar_makes_a_live_record_unproven(
        self, cfg: PodConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard's load-bearing leg, exercised the way the live pod check is.

        Deleting only the ``.start`` file leaves a pid that the service manager
        still agrees with, so pid equality alone would attest. It must not: an
        absent token is "unproven", and the mint refuses.
        """
        live = os.getpid()
        home = cfg.home_dir("demo")
        home.mkdir(parents=True, exist_ok=True)
        (home / ".local_secret").write_text("must-not-leave-this-process")
        path = _write_record(cfg, "demo", 7999, f"{live}\n", _live_start_token())
        monkeypatch.setattr(rt, "main_pid", lambda c, n: live)
        monkeypatch.setattr(rt, "derive_port", lambda c, n: 7999)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: False)
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_POD

        path.with_suffix(".start").unlink()
        monkeypatch.setattr(
            rt,
            "loopback_urlopen",
            lambda *a, **k: pytest.fail("the secret must not be sent"),
        )
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_UNPROVEN
        with pytest.raises(rt.PodOwnershipUnproven):
            rt.mint_token(cfg, "demo", "1h")

    def test_a_two_line_record_from_an_intermediate_build_is_unproven(
        self, cfg: PodConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pid file carrying its own token is no longer where the reader looks.

        Such a file exists only in a home written by an intermediate build. It
        must fail CLOSED -- and read as "record present, no start identity" so the
        operator is told to rebuild the worktree rather than to restart.
        """
        live = os.getpid()
        _write_record(cfg, "demo", 7999, f"{live}\n{_live_start_token()}\n")
        monkeypatch.setattr(rt, "main_pid", lambda c, n: live)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: False)
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_UNPROVEN
        assert rt._unproven_remedy(cfg, "demo", 7999).startswith("The record names a pid")


class TestPodUnprovenRemediation:
    """A refusal that prescribes the wrong fix is worse than a bare refusal.

    ``pod api`` is driven by agents, so the remedy line is acted on literally. A
    record with no start identity cannot be fixed by a restart -- the pod's
    gateway is its worktree's own venv binary, so a checkout that predates the
    sidecar writes no token however many times it is restarted.
    """

    @pytest.fixture
    def cfg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PodConfig:
        monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
        monkeypatch.setattr(rt, "IS_POSIX", True)
        return PodConfig.load()

    def test_no_record_at_all_is_told_to_restart(self, cfg: PodConfig) -> None:
        remedy = rt._unproven_remedy(cfg, "demo", 7999)
        assert "restarting it is what re-establishes the proof" in remedy
        assert "kirocrew pod down demo && kirocrew pod up demo" in remedy
        assert "provision" not in remedy

    def test_a_stale_generation_is_told_to_restart(self, cfg: PodConfig) -> None:
        """A token that simply does not match IS crash residue: restart fixes it."""
        _write_record(cfg, "demo", 7999, f"{os.getpid()}\n", "some-earlier-generation")
        remedy = rt._unproven_remedy(cfg, "demo", 7999)
        assert "restarting it is what re-establishes the proof" in remedy
        assert "provision" not in remedy

    def test_a_record_with_no_start_identity_is_told_to_rebuild(self, cfg: PodConfig) -> None:
        """The case the old single message got wrong."""
        _write_record(cfg, "demo", 7999, f"{os.getpid()}\n")
        remedy = rt._unproven_remedy(cfg, "demo", 7999)
        assert "re-establishes the proof" not in remedy
        assert "kirocrew pod provision demo" in remedy
        assert "worktree" in remedy

    def test_the_mint_refusal_carries_the_matching_remedy(
        self, cfg: PodConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The split reaches the message a caller actually sees."""
        home = cfg.home_dir("demo")
        home.mkdir(parents=True, exist_ok=True)
        (home / ".local_secret").write_text("must-not-leave-this-process")
        _write_record(cfg, "demo", 7999, f"{os.getpid()}\n")
        monkeypatch.setattr(rt, "derive_port", lambda c, n: 7999)
        monkeypatch.setattr(rt, "main_pid", lambda c, n: os.getpid())
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: False)
        monkeypatch.setattr(
            rt,
            "loopback_urlopen",
            lambda *a, **k: pytest.fail("the secret must not be sent"),
        )
        with pytest.raises(rt.PodOwnershipUnproven) as excinfo:
            rt.mint_token(cfg, "demo", "1h")
        message = str(excinfo.value)
        assert "kirocrew pod provision demo" in message
        assert "re-establishes the proof" not in message
        assert "must-not-leave-this-process" not in message


class TestPodPidAttestation:
    def test_a_fresh_pid_record_attests_without_any_listener_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A host with no ``lsof`` is a supported platform, not a broken one.

        The record proves its own freshness, so pid/MainPID agreement stands on
        its own and the lookup is never even attempted. Withholding here would
        deny the credential to every healthy pod on a minimal Linux box forever.
        """
        monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
        monkeypatch.setattr(rt, "IS_POSIX", True)
        live = os.getpid()
        monkeypatch.setattr(rt, "main_pid", lambda cfg, name: live)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: False)
        monkeypatch.setattr(
            rt,
            "find_port_listeners",
            lambda port: pytest.fail("an unavailable listener tool must not be called"),
        )
        cfg = PodConfig.load()
        _write_record(cfg, "demo", 7999, f"{live}\n", _live_start_token())
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_POD

    def test_a_pid_record_with_no_start_time_line_is_unproven(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The historical one-line format cannot prove it is not a crash leftover."""
        monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
        monkeypatch.setattr(rt, "IS_POSIX", True)
        live = os.getpid()
        monkeypatch.setattr(rt, "main_pid", lambda cfg, name: live)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: False)
        cfg = PodConfig.load()
        _write_record(cfg, "demo", 7999, f"{live}\n")
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_UNPROVEN

    def test_a_listener_tool_corroborates_the_pid_agreement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
        monkeypatch.setattr(rt, "IS_POSIX", True)
        _pin_start_token(monkeypatch, 4242, _START_TOKEN_FOR_4242)
        monkeypatch.setattr(rt, "main_pid", lambda cfg, name: 4242)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
        monkeypatch.setattr(
            rt,
            "find_port_listeners",
            lambda port: [platform_compat.PortListener(4242, "127.0.0.1", "4")],
        )
        cfg = PodConfig.load()
        _write_record(cfg, "demo", 7999, "4242\n", _START_TOKEN_FOR_4242)
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_POD

    def test_listener_attribution_alone_never_vouches_for_a_pod(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
        monkeypatch.setattr(rt, "IS_POSIX", True)
        monkeypatch.setattr(rt, "main_pid", lambda cfg, name: 4242)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
        monkeypatch.setattr(
            rt,
            "find_port_listeners",
            lambda port: [platform_compat.PortListener(4242, "127.0.0.1", "4")],
        )
        assert rt.port_owner(PodConfig.load(), "demo", 7999) == rt.OWNER_UNPROVEN

    def test_a_listener_disagreeing_with_the_attested_pid_is_foreign(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
        monkeypatch.setattr(rt, "IS_POSIX", True)
        _pin_start_token(monkeypatch, 4242, _START_TOKEN_FOR_4242)
        monkeypatch.setattr(rt, "main_pid", lambda cfg, name: 4242)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
        monkeypatch.setattr(
            rt,
            "find_port_listeners",
            lambda port: [platform_compat.PortListener(9999, "127.0.0.1", "4")],
        )
        cfg = PodConfig.load()
        _write_record(cfg, "demo", 7999, "4242\n", _START_TOKEN_FOR_4242)
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_FOREIGN

    @pytest.mark.parametrize("recorded", ["", "not-a-pid", "4241", "9" * 65])
    def test_missing_malformed_or_stale_records_are_unproven(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str
    ) -> None:
        monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
        monkeypatch.setattr(rt, "IS_POSIX", True)
        _pin_start_token(monkeypatch, 4242, _START_TOKEN_FOR_4242)
        monkeypatch.setattr(rt, "main_pid", lambda cfg, name: 4242)
        monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: False)
        cfg = PodConfig.load()
        if recorded:
            _write_record(cfg, "demo", 7999, f"{recorded}\n", _START_TOKEN_FOR_4242)
        assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_UNPROVEN

    def test_non_posix_stays_unproven_without_touching_host_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rt, "IS_POSIX", False)
        monkeypatch.setattr(rt, "main_pid", lambda cfg, name: pytest.fail("must not run"))
        assert rt.port_owner(PodConfig.load(), "demo", 7999) == rt.OWNER_UNPROVEN


class TestPodApiSecurityPosture:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/sessions?token=caller-secret",
            "sessions?limit=2&%74oken=caller-secret",
            "http://127.0.0.1:7811/api/health?token=caller-secret",
            "/api/health?token",
        ],
    )
    def test_caller_tokens_are_refused_without_disclosure(
        self, path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rt, "is_active", lambda *_: pytest.fail("validated too late"))
        with pytest.raises(rt.PodError) as excinfo:
            rt.pod_api(PodConfig.load(), "wt", "GET", path)
        message = str(excinfo.value)
        assert "token" in message
        assert "caller-secret" not in message
        assert "http://" not in message

    def test_malformed_credential_url_is_never_echoed(self) -> None:
        supplied = "http://[bad/api/health?token=caller-secret"
        with pytest.raises(rt.PodError) as excinfo:
            rt.api_path(supplied)
        message = str(excinfo.value)
        assert supplied not in message
        assert "caller-secret" not in message

    def test_transport_errors_never_quote_the_authenticated_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "s3cret-pod-token"
        monkeypatch.setattr(rt, "is_active", lambda cfg, name: True)
        monkeypatch.setattr(rt, "derive_port", lambda cfg, name: 7999)
        monkeypatch.setattr(rt, "mint_token", lambda cfg, name: secret)

        def _fail(req: Any, timeout: int = 0) -> None:
            raise OSError(f"connect failed for {req.full_url}")

        monkeypatch.setattr(rt, "loopback_urlopen", _fail)
        with pytest.raises(rt.PodError) as excinfo:
            rt.pod_api(PodConfig.load(), "wt", "GET", "sessions")
        rendered = str(excinfo.value)
        assert secret not in rendered
        assert "?token=" not in rendered

    def test_response_echoes_cannot_print_the_minted_token(
        self, stub_gateway: _RecordingServer
    ) -> None:
        status, raw = rt.pod_api(PodConfig.load(), "wt", "GET", "echo-token")
        assert status == 200
        assert "fixture-token" not in raw
        assert "<token>" in raw

    def test_json_escaped_keys_and_all_token_forms_are_scrubbed_before_cli_output(
        self,
        stub_gateway: _RecordingServer,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        token = "fixture token/+"
        stub_gateway.token = token
        events: list[tuple[tuple[object, ...], dict[str, object]]] = []
        monkeypatch.setattr(rt, "mint_token", lambda cfg, name, ttl="2h": token)
        monkeypatch.setattr(
            pod_cli,
            "_audit",
            lambda *args, **kwargs: events.append((args, kwargs)),
        )
        pod_cli._api(
            PodConfig.load(),
            argparse.Namespace(
                name="wt",
                method="GET",
                path="echo-escaped-token",
                data="",
                allow_write=False,
            ),
        )
        output = capsys.readouterr().out
        document = json.loads(output)
        assert document["body"] == {
            "<token>": {
                "raw": "<token>",
                "url": "<token>",
                "plus": "<token>",
            }
        }
        forms = (
            token,
            urllib.parse.quote(token, safe=""),
            urllib.parse.quote_plus(token, safe=""),
        )
        rendered = output + repr(events)
        assert all(form not in rendered for form in forms)

    def test_non_json_fallback_scrubs_all_token_encodings(self) -> None:
        token = "fixture token/+"
        forms = (
            token,
            urllib.parse.quote(token, safe=""),
            urllib.parse.quote_plus(token, safe=""),
            "".join(f"\\u{ord(char):04x}" for char in token),
        )
        scrubbed = rt._scrub_token("not-json: " + " | ".join(forms), token)
        assert all(form not in scrubbed for form in forms)
        assert scrubbed.count("<token>") == len(forms)


class TestPodApiHttp:
    def test_timeout_is_not_part_of_the_public_runtime_surface(self) -> None:
        assert "timeout" not in inspect.signature(rt.pod_api).parameters

    def test_authenticates_with_query_token_not_bearer_header(
        self, stub_gateway: _RecordingServer
    ) -> None:
        status, raw = rt.pod_api(PodConfig.load(), "wt", "GET", "sessions")
        assert status == 200
        assert json.loads(raw) == {"ok": True, "sent": ""}
        assert stub_gateway.seen[-1]["target"] == "/api/sessions?token=fixture-token"
        assert stub_gateway.seen[-1]["authorization"] == ""

    def test_preserves_existing_query_parameters(self, stub_gateway: _RecordingServer) -> None:
        status, _ = rt.pod_api(PodConfig.load(), "wt", "GET", "sessions?limit=2&kind=chat")
        assert status == 200
        assert stub_gateway.seen[-1]["target"] == (
            "/api/sessions?limit=2&kind=chat&token=fixture-token"
        )

    def test_get_and_head_are_safe_by_default(self, stub_gateway: _RecordingServer) -> None:
        assert rt.pod_api(PodConfig.load(), "wt", "GET", "health")[0] == 200
        assert rt.pod_api(PodConfig.load(), "wt", "HEAD", "health")[0] == 200

    def test_write_is_refused_before_host_access_without_allow_write(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rt, "is_active", lambda *_: pytest.fail("guard ran too late"))
        with pytest.raises(rt.PodError, match="--allow-write"):
            rt.pod_api(PodConfig.load(), "wt", "POST", "config", data="{}")

    def test_write_succeeds_with_allow_write(self, stub_gateway: _RecordingServer) -> None:
        status, raw = rt.pod_api(
            PodConfig.load(),
            "wt",
            "POST",
            "config",
            data='{"key":"x"}',
            allow_write=True,
        )
        assert status == 200
        assert json.loads(raw)["sent"] == '{"key":"x"}'
        assert stub_gateway.seen[-1]["method"] == "POST"

    def test_non_2xx_response_and_body_are_returned(self, stub_gateway: _RecordingServer) -> None:
        status, raw = rt.pod_api(PodConfig.load(), "wt", "GET", "boom")
        assert status == 503
        assert json.loads(raw) == {"error": "unavailable"}


class TestPodApiBoundedReads:
    def test_body_one_byte_over_the_cap_is_refused(self) -> None:
        class _Big:
            def read(self, size: int) -> bytes:
                return b"x" * size

        with pytest.raises(rt.PodError, match=str(rt.API_BODY_MAX_BYTES)):
            rt._read_capped(_Big(), "GET", "/api/huge", "wt")

    def test_truncated_body_is_reported_without_a_traceback(self) -> None:
        class _Truncated:
            def read(self, size: int) -> bytes:
                raise http.client.IncompleteRead(b"partial", 99)

        with pytest.raises(rt.PodError, match="response body could not be read"):
            rt._read_capped(_Truncated(), "GET", "/api/items", "wt")

    def test_body_at_the_cap_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rt, "API_BODY_MAX_BYTES", 4)

        class _Exact:
            def read(self, size: int) -> bytes:
                return b"four"

        assert rt._read_capped(_Exact(), "GET", "/api/items", "wt") == "four"


class TestPodApiAuditAndCli:
    @staticmethod
    def _args(path: str, method: str = "GET", **kwargs: object) -> argparse.Namespace:
        return argparse.Namespace(
            name="wt",
            method=method,
            path=path,
            data=kwargs.get("data", ""),
            allow_write=kwargs.get("allow_write", False),
        )

    def test_refused_caller_token_never_reaches_audit_or_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        events: list[tuple[tuple[object, ...], dict[str, object]]] = []
        monkeypatch.setattr(
            pod_cli,
            "_audit",
            lambda *args, **kwargs: events.append((args, kwargs)),
        )
        secret = "caller-secret"
        with pytest.raises(SystemExit) as excinfo:
            pod_cli._api(
                PodConfig.load(),
                self._args(f"sessions?token={secret}"),
            )
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        document = json.loads(captured.out)
        rendered = captured.out + captured.err + repr(events)
        assert set(document) == {"name", "method", "path", "status", "ok", "body"}
        assert document["path"] == "/api/<invalid>"
        assert document["status"] == 0 and document["ok"] is False
        assert secret not in rendered
        assert "token=" not in rendered

    @pytest.mark.parametrize(
        ("escape_path", "name", "method", "path", "allow_write"),
        [
            ("name-validation", "bad/name", "GET", "sessions", False),
            ("path-validation", "wt", "GET", "sessions?token=caller-secret", False),
            ("method-validation", "wt", "OPTIONS", "sessions", False),
            ("write-refusal", "wt", "POST", "sessions", False),
            ("pod-not-running", "wt", "GET", "sessions", False),
            ("port-resolution", "wt", "GET", "sessions", False),
            ("ownership-token-refusal", "wt", "GET", "sessions", False),
            ("transport", "wt", "GET", "sessions", False),
            ("read-cap", "wt", "GET", "sessions", False),
            ("unexpected-runtime-error", "wt", "GET", "sessions", False),
        ],
    )
    def test_every_pre_envelope_escape_path_prints_fixed_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        escape_path: str,
        name: str,
        method: str,
        path: str,
        allow_write: bool,
    ) -> None:
        class _OversizedResponse:
            status = 200

            def __enter__(self) -> _OversizedResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self, size: int) -> bytes:
                return b"x" * size

        monkeypatch.setattr(pod_cli, "_audit", lambda *args, **kwargs: None)
        if escape_path == "pod-not-running":
            monkeypatch.setattr(rt, "is_active", lambda *args: False)
        elif escape_path == "port-resolution":
            monkeypatch.setattr(rt, "is_active", lambda *args: True)
            monkeypatch.setattr(
                rt,
                "derive_port",
                lambda *args: (_ for _ in ()).throw(rt.PodError("port resolution failed")),
            )
        elif escape_path == "ownership-token-refusal":
            monkeypatch.setattr(rt, "is_active", lambda *args: True)
            monkeypatch.setattr(rt, "derive_port", lambda *args: 7999)
            monkeypatch.setattr(
                rt,
                "mint_token",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    rt.PodOwnershipUnproven("ownership unproven")
                ),
            )
        elif escape_path == "transport":
            monkeypatch.setattr(rt, "is_active", lambda *args: True)
            monkeypatch.setattr(rt, "derive_port", lambda *args: 7999)
            monkeypatch.setattr(rt, "mint_token", lambda *args, **kwargs: "fixture-token")
            monkeypatch.setattr(
                rt,
                "loopback_urlopen",
                lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("closed")),
            )
        elif escape_path == "read-cap":
            monkeypatch.setattr(rt, "is_active", lambda *args: True)
            monkeypatch.setattr(rt, "derive_port", lambda *args: 7999)
            monkeypatch.setattr(rt, "mint_token", lambda *args, **kwargs: "fixture-token")
            monkeypatch.setattr(rt, "API_BODY_MAX_BYTES", 4)
            monkeypatch.setattr(
                rt, "loopback_urlopen", lambda *args, **kwargs: _OversizedResponse()
            )
        elif escape_path == "unexpected-runtime-error":
            monkeypatch.setattr(
                rt,
                "pod_api",
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("internal detail")),
            )

        args = self._args(path, method=method, allow_write=allow_write)
        args.name = name
        with pytest.raises(SystemExit) as excinfo:
            pod_cli._api(PodConfig.load(), args)

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert captured.err == ""
        document = json.loads(captured.out)
        assert set(document) == {"name", "method", "path", "status", "ok", "body"}
        assert document["status"] == 0
        assert document["ok"] is False

    def test_transport_failure_prints_fixed_json_to_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        events: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def _fail(*args: object, **kwargs: object) -> tuple[int, str]:
            raise rt.PodError("transport failed")

        monkeypatch.setattr(rt, "pod_api", _fail)
        monkeypatch.setattr(
            pod_cli,
            "_audit",
            lambda *args, **kwargs: events.append((args, kwargs)),
        )
        with pytest.raises(SystemExit) as excinfo:
            pod_cli._api(PodConfig.load(), self._args("sessions"))
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert captured.err == ""
        assert json.loads(captured.out) == {
            "name": "wt",
            "method": "GET",
            "path": "/api/sessions",
            "status": 0,
            "ok": False,
            "body": "transport failed",
        }
        assert events

    def test_success_prints_fixed_json_and_audits_credential_free_target(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        events: list[tuple[object, ...]] = []
        monkeypatch.setattr(rt, "pod_api", lambda *args, **kwargs: (200, '{"ok":true}'))
        monkeypatch.setattr(pod_cli, "_audit", lambda *args, **kwargs: events.append(args))
        pod_cli._api(PodConfig.load(), self._args("sessions?limit=2"))
        document = json.loads(capsys.readouterr().out)
        assert document == {
            "name": "wt",
            "method": "GET",
            "path": "/api/sessions?limit=2",
            "status": 200,
            "ok": True,
            "body": {"ok": True},
        }
        assert events
        rendered = repr(events)
        assert "pod.api" in rendered
        assert "token=" not in rendered

    def test_non_2xx_prints_the_same_shape_and_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(rt, "pod_api", lambda *args, **kwargs: (503, "unavailable"))
        monkeypatch.setattr(pod_cli, "_audit", lambda *args, **kwargs: None)
        with pytest.raises(SystemExit) as excinfo:
            pod_cli._api(PodConfig.load(), self._args("boom"))
        assert excinfo.value.code == 1
        document = json.loads(capsys.readouterr().out)
        assert document["status"] == 503
        assert document["ok"] is False
        assert document["body"] == "unavailable"

    def test_lowercase_method_is_normalized(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[str] = []

        def _call(cfg: PodConfig, name: str, method: str, path: str, **kwargs: object):
            calls.append(method)
            return 200, ""

        monkeypatch.setattr(rt, "pod_api", _call)
        monkeypatch.setattr(pod_cli, "_audit", lambda *args, **kwargs: None)
        pod_cli._api(PodConfig.load(), self._args("health", method="get"))
        assert calls == ["GET"]
        assert json.loads(capsys.readouterr().out)["method"] == "GET"

    @pytest.mark.parametrize("depth", [2_000, 20_000])
    def test_a_response_too_deep_for_json_still_prints_fixed_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], depth: int
    ) -> None:
        """A body that exhausts the recursion budget must not replace the envelope.

        Both halves of the round trip recurse once per nesting level, and which
        one gives way depends on how much stack is left — measured here, depth
        20,000 fails to decode while depth 2,000 decodes and then fails to
        ENCODE. So this asserts the invariant that holds either way rather than
        pinning a threshold that shifts with the interpreter and the call depth.
        """
        raw = "[" * depth + "]" * depth
        monkeypatch.setattr(rt, "pod_api", lambda *args, **kwargs: (200, raw))
        monkeypatch.setattr(pod_cli, "_audit", lambda *args, **kwargs: None)

        pod_cli._api(PodConfig.load(), self._args("sessions"))

        captured = capsys.readouterr()
        assert captured.err == ""
        document = json.loads(captured.out)
        assert set(document) == {"name", "method", "path", "status", "ok", "body"}
        # The request itself succeeded, so its status survives whichever leg gave way.
        assert document["status"] == 200
        assert document["ok"] is True
        assert document["body"] in (raw, "<body omitted: not serializable>")

    def test_a_decode_failure_no_one_enumerated_degrades_to_text(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The decode boundary must fail safe on ANY exception, not a listed set.

        RecursionError is raised here directly because it is the reachable case
        that a `except ValueError` decode handler lets through.
        """

        def _explode(*args: object, **kwargs: object) -> object:
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(pod_cli.json, "loads", _explode)
        assert pod_cli._api_body('{"ok": true}') == '{"ok": true}'

    def test_the_envelope_degrades_a_body_it_cannot_serialize(self) -> None:
        """The render is the command's last exit, so it must not raise either.

        Driven with an unserializable object because that failure is
        deterministic at any stack depth, unlike a depth-dependent one.
        """
        rendered = pod_cli._api_envelope("wt", "GET", "/api/sessions", 200, True, object())

        document = json.loads(rendered)
        assert set(document) == {"name", "method", "path", "status", "ok", "body"}
        assert document["status"] == 200
        assert document["body"] == "<body omitted: not serializable>"
