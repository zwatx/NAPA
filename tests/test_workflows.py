from __future__ import annotations

from napa_vision.workflows import calibration_main


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
