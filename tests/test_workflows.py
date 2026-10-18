from __future__ import annotations

import json
from pathlib import Path

from napa_vision.workflows import calibration_main, layout_main


def _layout(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "field": {"length": 16.5, "width": 8.2},
                "tags": [
                    {
                        "ID": 1,
                        "pose": {
                            "translation": {"x": 1.0, "y": 2.0, "z": 1.5},
                            "rotation": {"quaternion": {"W": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0}},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_layout_workflow_reports_provenance_and_production_gate(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    report = tmp_path / "report.json"
    _layout(path)
    assert layout_main([str(path), "--report", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["tag_ids"] == [1]
    assert payload["production_ready"] is False

    assert layout_main([str(path), "--require-production"]) == 2
    assert layout_main([str(path), "--official-source", "WPILib verified file", "--require-production"]) == 0


def test_calibration_workflow_reports_missing_directory() -> None:
    assert (
        calibration_main(
            [
                "/does/not/exist",
                "--output",
                "unused.json",
                "--columns",
                "9",
                "--rows",
                "6",
                "--square-size-m",
                "0.025",
            ]
        )
        == 2
    )
