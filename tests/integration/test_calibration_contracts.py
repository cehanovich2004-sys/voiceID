"""Integration checks for the core-only Phase 5B calibration contracts."""

from __future__ import annotations

import importlib.resources
import os
import subprocess
import sys
from pathlib import Path


def test_installed_calibration_import_is_core_only_without_network(
    tmp_path: Path,
) -> None:
    before = set(os.listdir(tmp_path))

    script = """
import importlib.abc
import importlib.resources
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

from voiceid.calibration import (
    CALIBRATION_CONTRACT_VERSION,
    CALIBRATION_PROTOCOL_IDENTIFIER,
    CalibrationExperimentPlan,
    validate_calibration_experiment_plan,
)

assert CALIBRATION_PROTOCOL_IDENTIFIER == 'phase5b-experimental-calibration-protocol-v1'
assert CALIBRATION_CONTRACT_VERSION == 'phase5b-experiment-contracts-v1'
assert CalibrationExperimentPlan.__name__ == 'CalibrationExperimentPlan'
assert callable(validate_calibration_experiment_plan)
assert importlib.resources.files('voiceid').joinpath('py.typed').is_file()
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


def test_py_typed_is_present_for_packaged_voiceid() -> None:
    assert importlib.resources.files("voiceid").joinpath("py.typed").is_file()
