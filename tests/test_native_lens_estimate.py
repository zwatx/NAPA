"""Known-geometry checks for the radial lens estimated from a frame-cut tag's own sides."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("scalar", [False, True])
def test_lens_estimate(tmp_path: Path, scalar: bool) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native kernel checks require cc")
    root = Path(__file__).resolve().parents[1]
    if not (root / "src/napa_vision/_native/napa_core.c").is_file():
        pytest.skip("native kernel checks require a source checkout")
    executable = tmp_path / "lens-estimate"
    command = [compiler, "-std=c99", "-O2", "-fno-wrapv", "-pthread"]
    if scalar:
        command.append("-DNAPA_NO_SIMD")
    command.extend([str(root / "tests/native_lens_estimate.c"), "-lm", "-o", str(executable)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "shared lens estimate known-geometry cases passed" in result.stdout
