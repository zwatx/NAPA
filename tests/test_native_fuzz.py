"""Run the C core's fuzz target as a short random driver (tests/fuzz/napa_fuzz.c), with sanitizers when available."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SANITIZE = ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer"]


@pytest.mark.parametrize("scalar", [False, True], ids=["native", "scalar"])
def test_fuzz_driver(tmp_path: Path, scalar: bool) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("the fuzz driver requires a cc compiler")
    if not (ROOT / "src/napa_vision/_native/napa_core.c").is_file():
        pytest.skip("the fuzz driver requires a source checkout")
    executable = tmp_path / "napa_fuzz"
    sources = [str(ROOT / "tests/fuzz/napa_fuzz.c"), str(ROOT / "src/napa_vision/_native/napa_core.c")]
    command = [compiler, "-std=c99", "-g", "-O1", "-pthread", "-fno-wrapv", *(["-DNAPA_NO_SIMD"] if scalar else [])]
    # Sanitizers where the toolchain has them (CI's native-sanitizers job always does); a plain build still checks
    # the API contract. A toolchain without them fails on a trivial program too, so the probe tells the two apart.
    probe = tmp_path / "probe.c"
    probe.write_text("int main(void) { return 0; }\n")
    sanitizers = subprocess.run([compiler, *SANITIZE, str(probe), "-o", str(tmp_path / "probe")], capture_output=True)
    flags = SANITIZE if sanitizers.returncode == 0 else []
    built = subprocess.run([*command, *flags, *sources, "-lm", "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr[-4000:]
    options = {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1", "UBSAN_OPTIONS": "print_stacktrace=1"}
    result = subprocess.run([str(executable), "150"], capture_output=True, text=True, env={**os.environ, **options})
    assert result.returncode == 0, result.stderr[-4000:]
    assert "150 random fuzz inputs passed" in result.stdout
    assert ", 0 detections" not in result.stdout  # drawn tags must reach decoding, not just the search
