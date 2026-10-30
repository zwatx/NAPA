"""Record the source/build inputs for a fresh replay, without changing saved measurements."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(os.environ.get("NAPA_BENCH_OUTPUT_DIR", ROOT / "work/optimization/reproduction"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data = {
        "base_commit": "674c91b",
        "candidate_source_sha256": sha(ROOT / "src/napa_vision/_native/napa_core.c"),
        "candidate_binary_sha256": sha(ROOT / "work/optimization/libnapa_core_candidate.so"),
        "baseline_source_sha256": sha(ROOT / "work/optimization/baseline.c"),
        "baseline_binary_sha256": sha(ROOT / "work/optimization/baseline.so"),
        "pre_terminal_guard_source_sha256": sha(Path(__file__).with_name("diagnostic_napa_core.c")),
        "pre_terminal_guard_binary_sha256": sha(ROOT / "work/optimization/pre-roi.so"),
        "unpacked_source_sha256": sha(Path(__file__).with_name("unpacked_napa_core.c")),
        "unpacked_binary_sha256": sha(ROOT / "work/optimization/unpacked.so"),
        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        "resource_frames_sha256": {
            name: sha(ROOT / f"work/frames/{name}.npz") for name in ("640x480", "1280x720", "1920x1080")
        },
        "note": (
            "Fresh replay: input JSONs record actual loaded library hashes. "
            "Intermediate source snapshots are used only for exact output comparisons."
        ),
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "source_manifest.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
