"""Tests for the MCP-launcher orphan SUBTREE reap.

Regression cover for a leak where 112 processes (15.2 GB RSS) survived 23 days
of sweeps on one host. The sweep reclaimed the marked launcher at the top of an
orphaned tree and relied on surviving children reparenting to init to become
candidates themselves on a later pass. That fallback breaks on an UNMARKED
intermediate: it is a candidate but not sweepable, so it lives forever AND hides
its own marked children behind a ppid that is not init, where
``_our_orphan_pids`` never enumerates them.

Observed shape (Amazon Builder Toolbox, but any wrapper execing a resolved
binary produces it)::

    aim mcp start-server npm:@playwright/mcp   <- marked, swept
      -> <toolbox>/aim mcp start-server ...    <- marked, swept
        -> node .../3P/bin/playwright-mcp      <- UNMARKED, leaked
          -> npm exec @playwright/mcp@latest   <- marked but unreachable
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX process-management semantics only; see issue #2041"
)

# Root of the orphaned tree: a marked MCP launcher that passes the sweep gate.
_ROOT = 500
_ROOT_CMDLINE = b"python3\x00kirocrew_sandbox_x.py"

# The leaked chain hanging off it.
_UNMARKED_INTERMEDIATE = 501
_MARKED_LEAF = 502

_CMDLINES = {
    _ROOT: _ROOT_CMDLINE,
    # No launcher marker in this argv at all -- this is the process that broke
    # the "reclaimed on a subsequent sweep" fallback.
    _UNMARKED_INTERMEDIATE: b"node\x00/home/u/.aim/mcp-servers/3P/bin/playwright-mcp",
    _MARKED_LEAF: b"npm\x00exec\x00@playwright/mcp@latest",
}


def _fake_read_bytes(self: Path) -> bytes:
    """Serve /proc/<pid>/cmdline reads from _CMDLINES by path."""
    parts = self.parts
    for pid, cmdline in _CMDLINES.items():
        if str(pid) in parts:
            return cmdline
    return _ROOT_CMDLINE


@_POSIX_ONLY
class TestOrphanMcpSubtreeReap:
    """kill_orphan_mcps must reap the whole orphaned launcher tree."""

    def _run(
        self,
        descendants: list[int],
        *,
        marked: set[int],
        pgid: int = _ROOT,
        my_pgid: int = 1000,
    ) -> list[int]:
        """Run the sweep over _ROOT and return the PIDs killed in the subtree."""
        from kiro_crew.session_pid import kill_orphan_mcps

        subtree_kills: list[int] = []
        with (
            patch("os.getpgrp", return_value=my_pgid),
            patch("os.getpgid", return_value=pgid),
            patch("os.killpg"),
            patch("os.kill"),
            patch("os.getpid", return_value=1),
            patch("kiro_crew.session_pid.sys") as mock_sys,
            patch.object(Path, "read_bytes", _fake_read_bytes),
            patch("kiro_crew.acp.client._get_child_pids", return_value=descendants),
            patch(
                "kiro_crew.session_pid._env_has_kirocrew_marker",
                side_effect=lambda pid: pid in marked,
            ),
            patch(
                "kiro_crew.session_pid.platform_compat.kill_pid",
                side_effect=lambda pid, _sig: subtree_kills.append(pid),
            ),
        ):
            mock_sys.platform = "linux"
            kill_orphan_mcps([_ROOT])
        return subtree_kills

    def test_marked_leaf_behind_unmarked_intermediate_is_reaped(self) -> None:
        """The regression: a marked leaf hidden behind an unmarked parent dies."""
        killed = self._run(
            [_UNMARKED_INTERMEDIATE, _MARKED_LEAF],
            marked={_ROOT, _UNMARKED_INTERMEDIATE, _MARKED_LEAF},
        )

        assert _MARKED_LEAF in killed, "leaked leaf must not survive the sweep"
        assert _UNMARKED_INTERMEDIATE in killed, "the blocking intermediate must die too"

    def test_kills_leaf_first(self) -> None:
        """Every process dies before its parent, so none reparents mid-kill."""
        killed = self._run(
            [_UNMARKED_INTERMEDIATE, _MARKED_LEAF],
            marked={_ROOT, _UNMARKED_INTERMEDIATE, _MARKED_LEAF},
        )

        assert killed.index(_MARKED_LEAF) < killed.index(_UNMARKED_INTERMEDIATE)

    def test_descendant_without_env_marker_is_spared(self) -> None:
        """Positive identity per member: no KIROCREW_SPAWNED, no kill.

        The root passing the sweep gate does not license killing arbitrary
        descendants -- an unrelated process that merely landed in the tree keeps
        running.
        """
        killed = self._run(
            [_UNMARKED_INTERMEDIATE, _MARKED_LEAF],
            marked={_ROOT, _MARKED_LEAF},  # intermediate is NOT Kiro-Crew-spawned
        )

        assert _UNMARKED_INTERMEDIATE not in killed
        assert _MARKED_LEAF in killed

    def test_peer_gateway_descendant_is_spared(self) -> None:
        """A dev pod / peer gateway under the tree survives (_GATEWAY_MARKERS)."""
        gateway_pid = 503
        _CMDLINES[gateway_pid] = b"python3\x00-m\x00kiro_crew.cli\x00gateway"
        try:
            killed = self._run(
                [gateway_pid, _MARKED_LEAF],
                marked={_ROOT, gateway_pid, _MARKED_LEAF},
            )
        finally:
            del _CMDLINES[gateway_pid]

        assert gateway_pid not in killed, "never signal a peer gateway"
        assert _MARKED_LEAF in killed

    def test_root_is_not_signalled_twice(self) -> None:
        """The root already took killpg/os.kill; the subtree walk skips it."""
        killed = self._run(
            [_ROOT, _MARKED_LEAF],  # a PID cycle would surface the root here
            marked={_ROOT, _MARKED_LEAF},
        )

        assert _ROOT not in killed

    def test_subtree_counts_against_the_global_kill_cap(self) -> None:
        """Subtree members share _ORPHAN_SWEEP_MAX_KILLS with roots."""
        from kiro_crew.session_pid import _ORPHAN_SWEEP_MAX_KILLS

        descendants = list(range(600, 600 + _ORPHAN_SWEEP_MAX_KILLS + 10))
        for pid in descendants:
            _CMDLINES[pid] = b"npm\x00exec\x00@playwright/mcp@latest"
        try:
            killed = self._run(descendants, marked={_ROOT, *descendants})
        finally:
            for pid in descendants:
                del _CMDLINES[pid]

        # The root consumed one kill, so the subtree may use at most cap - 1.
        assert len(killed) == _ORPHAN_SWEEP_MAX_KILLS - 1

    def test_vanished_descendant_is_skipped(self) -> None:
        """A PID that exits between enumeration and kill is not an error."""
        from kiro_crew.session_pid import kill_orphan_mcps

        def _raising_read(self: Path) -> bytes:
            if str(_MARKED_LEAF) in self.parts:
                raise ProcessLookupError
            return _ROOT_CMDLINE

        subtree_kills: list[int] = []
        with (
            patch("os.getpgrp", return_value=1000),
            patch("os.getpgid", return_value=_ROOT),
            patch("os.killpg"),
            patch("os.getpid", return_value=1),
            patch("kiro_crew.session_pid.sys") as mock_sys,
            patch.object(Path, "read_bytes", _raising_read),
            patch("kiro_crew.acp.client._get_child_pids", return_value=[_MARKED_LEAF]),
            patch("kiro_crew.session_pid._env_has_kirocrew_marker", return_value=True),
            patch(
                "kiro_crew.session_pid.platform_compat.kill_pid",
                side_effect=lambda pid, _sig: subtree_kills.append(pid),
            ),
        ):
            mock_sys.platform = "linux"
            killed = kill_orphan_mcps([_ROOT])

        assert subtree_kills == []
        assert killed == 1  # the root still counted

    def test_subtree_enumerated_before_root_dies(self) -> None:
        """Enumeration must precede the root signal, or the links are gone."""
        order: list[str] = []
        with (
            patch("os.getpgrp", return_value=1000),
            patch("os.getpgid", return_value=_ROOT),
            patch("os.killpg", side_effect=lambda *_a: order.append("killpg")),
            patch("os.getpid", return_value=1),
            patch("kiro_crew.session_pid.sys") as mock_sys,
            patch.object(Path, "read_bytes", _fake_read_bytes),
            patch(
                "kiro_crew.acp.client._get_child_pids",
                side_effect=lambda _pid: order.append("enumerate") or [],
            ),
            patch("kiro_crew.session_pid._env_has_kirocrew_marker", return_value=True),
            patch("kiro_crew.session_pid.platform_compat.kill_pid"),
        ):
            mock_sys.platform = "linux"
            kill_orphan_mcps_ = __import__(
                "kiro_crew.session_pid", fromlist=["kill_orphan_mcps"]
            ).kill_orphan_mcps
            kill_orphan_mcps_([_ROOT])

        assert order == ["enumerate", "killpg"]


@_POSIX_ONLY
class TestOrphanMcpSubtreeHelper:
    """Direct cover for _kill_orphan_mcp_descendants edge cases."""

    def test_zero_budget_kills_nothing(self) -> None:
        from kiro_crew.session_pid import _kill_orphan_mcp_descendants

        with patch("kiro_crew.session_pid.platform_compat.kill_pid") as mock_kill:
            killed = _kill_orphan_mcp_descendants([_MARKED_LEAF], root=_ROOT, budget=0)

        assert killed == 0
        mock_kill.assert_not_called()

    def test_empty_descendants_kills_nothing(self) -> None:
        from kiro_crew.session_pid import _kill_orphan_mcp_descendants

        with patch("kiro_crew.session_pid.platform_compat.kill_pid") as mock_kill:
            killed = _kill_orphan_mcp_descendants([], root=_ROOT, budget=10)

        assert killed == 0
        mock_kill.assert_not_called()

    def test_never_signals_pid_one_or_self(self) -> None:
        from kiro_crew.session_pid import _kill_orphan_mcp_descendants

        with (
            patch("os.getpid", return_value=4242),
            patch("os.getpgrp", return_value=9999),
            patch("kiro_crew.session_pid.platform_compat.kill_pid") as mock_kill,
        ):
            killed = _kill_orphan_mcp_descendants([1, 0, 4242, 9999], root=_ROOT, budget=10)

        assert killed == 0
        mock_kill.assert_not_called()

    def test_windows_is_a_noop(self) -> None:
        """taskkill /T already walked the tree when the session ended."""
        from kiro_crew.session_pid import _kill_orphan_mcp_descendants

        with (
            patch("kiro_crew.session_pid.platform_compat.IS_WINDOWS", True),
            patch("kiro_crew.session_pid.platform_compat.kill_pid") as mock_kill,
        ):
            killed = _kill_orphan_mcp_descendants([_MARKED_LEAF], root=_ROOT, budget=10)

        assert killed == 0
        mock_kill.assert_not_called()

    def test_emits_sel_audit_when_it_kills(self) -> None:
        from kiro_crew.session_pid import _kill_orphan_mcp_descendants

        with (
            patch("os.getpid", return_value=1),
            patch("os.getpgrp", return_value=1000),
            patch("kiro_crew.session_pid.sys") as mock_sys,
            patch.object(Path, "read_bytes", _fake_read_bytes),
            patch("kiro_crew.session_pid._env_has_kirocrew_marker", return_value=True),
            patch("kiro_crew.session_pid.platform_compat.kill_pid"),
            patch("kiro_crew.session_pid._sel_orphan_mcp_subtree_kill") as mock_sel,
        ):
            mock_sys.platform = "linux"
            killed = _kill_orphan_mcp_descendants([_MARKED_LEAF], root=_ROOT, budget=10)

        assert killed == 1
        mock_sel.assert_called_once_with(_ROOT, 1)

    def test_no_sel_audit_when_nothing_dies(self) -> None:
        from kiro_crew.session_pid import _kill_orphan_mcp_descendants

        with (
            patch("os.getpid", return_value=1),
            patch("os.getpgrp", return_value=1000),
            patch("kiro_crew.session_pid.sys") as mock_sys,
            patch.object(Path, "read_bytes", _fake_read_bytes),
            patch("kiro_crew.session_pid._env_has_kirocrew_marker", return_value=False),
            patch("kiro_crew.session_pid.platform_compat.kill_pid"),
            patch("kiro_crew.session_pid._sel_orphan_mcp_subtree_kill") as mock_sel,
        ):
            mock_sys.platform = "linux"
            killed = _kill_orphan_mcp_descendants([_MARKED_LEAF], root=_ROOT, budget=10)

        assert killed == 0
        mock_sel.assert_not_called()

    def test_kill_pid_failure_is_not_counted(self) -> None:
        from kiro_crew.session_pid import _kill_orphan_mcp_descendants

        with (
            patch("os.getpid", return_value=1),
            patch("os.getpgrp", return_value=1000),
            patch("kiro_crew.session_pid.sys") as mock_sys,
            patch.object(Path, "read_bytes", _fake_read_bytes),
            patch("kiro_crew.session_pid._env_has_kirocrew_marker", return_value=True),
            patch(
                "kiro_crew.session_pid.platform_compat.kill_pid",
                side_effect=ProcessLookupError,
            ),
        ):
            mock_sys.platform = "linux"
            killed = _kill_orphan_mcp_descendants([_MARKED_LEAF], root=_ROOT, budget=10)

        assert killed == 0

    def test_signal_passed_is_sigkill(self) -> None:
        from kiro_crew.session_pid import _kill_orphan_mcp_descendants

        with (
            patch("os.getpid", return_value=1),
            patch("os.getpgrp", return_value=1000),
            patch("kiro_crew.session_pid.sys") as mock_sys,
            patch.object(Path, "read_bytes", _fake_read_bytes),
            patch("kiro_crew.session_pid._env_has_kirocrew_marker", return_value=True),
            patch("kiro_crew.session_pid.platform_compat.kill_pid") as mock_kill,
        ):
            mock_sys.platform = "linux"
            _kill_orphan_mcp_descendants([_MARKED_LEAF], root=_ROOT, budget=10)

        assert mock_kill.call_args[0][0] == _MARKED_LEAF
        assert mock_kill.call_args[0][1] in (signal.SIGKILL, int(signal.SIGKILL))
