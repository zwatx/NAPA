"""Independent regression checks for native line fitting."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("scalar", [False, True])
def test_line_fit(tmp_path: Path, scalar: bool) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native kernel checks require cc")
    root = Path(__file__).resolve().parents[1]
    if not (root / "src/napa_vision/_native/napa_core.c").is_file():
        pytest.skip("native kernel checks require a source checkout")
    executable = tmp_path / "line-fit"
    command = [compiler, "-std=c99", "-O2", "-fno-wrapv", "-pthread"]
    if scalar:
        command.append("-DNAPA_NO_SIMD")
    command.extend([str(root / "tests/native_line_fit.c"), "-lm", "-o", str(executable)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "TLS independent fallback" in result.stdout
