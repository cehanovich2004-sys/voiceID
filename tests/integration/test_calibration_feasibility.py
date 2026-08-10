"""Integration checks for the opt-in feasibility probe boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_feasibility_import_is_core_only_without_network(tmp_path: Path) -> None:
    before = set(os.listdir(tmp_path))
    script = """
import importlib.abc
import socket
import sys

class BlockMlImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'torchaudio', 'speechbrain'}:
            raise ImportError(f'blocked optional ML dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockMlImports())

def fail_network(*args, **kwargs):
    raise RuntimeError('network access is forbidden')

socket.socket.connect = fail_network
socket.create_connection = fail_network

probe_socket = socket.socket()
try:
    try:
        probe_socket.connect(('127.0.0.1', 9))
    except RuntimeError as exc:
        assert str(exc) == 'network access is forbidden'
    else:
        raise AssertionError('socket.connect guard is inactive')
finally:
    probe_socket.close()

try:
    socket.create_connection(('127.0.0.1', 9), timeout=0.001)
except RuntimeError as exc:
    assert str(exc) == 'network access is forbidden'
else:
    raise AssertionError('socket.create_connection guard is inactive')

from voiceid.calibration.feasibility import (
    FEASIBILITY_LABEL_CRITERIA_VERSION,
    FeasibilityLabel,
    run_feasibility_probe,
)

assert FEASIBILITY_LABEL_CRITERIA_VERSION == 'phase5b-feasibility-label-v1'
assert FeasibilityLabel.PROMISING.value == 'PROMISING'
assert callable(run_feasibility_probe)
assert not {'torch', 'torchaudio', 'speechbrain'} & set(sys.modules)
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert set(os.listdir(tmp_path)) == before
