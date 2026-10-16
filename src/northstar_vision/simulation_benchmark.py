from __future__ import annotations

import platform
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .config import PlatformProfile
from .fiducials import OpenCVArucoFiducialDetector
from .service import NorthstarVisionService
from .simulation import RenderedFrame, SyntheticFrameSource
from .simulation_artifacts import SIMULATION_LIMITATIONS
from .simulation_metrics import metric_distribution
from .simulation_renderer import SyntheticAprilTagRenderer
from .simulation_scenarios import simulation_frames, synthetic_wall_layout
from .simulation_scoring import SimulationScoringPublisher
from .types import CameraPoseBounds


@dataclass(frozen=True, slots=True)
class SimulationCorpus:
    frames: tuple[RenderedFrame, ...]
    tag_layout: dict[int, np.ndarray]
    render_ms: tuple[float, ...]
    suite: str
    seed: int
    supersample: int
    profile_key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class SimulationRun:
    frames: tuple[RenderedFrame, ...]
    tag_layout: dict[int, np.ndarray]
    summary: dict[str, Any]
    corpus: SimulationCorpus | None = None


def _profile_key(profile: PlatformProfile) -> tuple[object, ...]:
    camera = profile.camera
    return (
        profile.name,
        camera.width,
        camera.height,
        profile.tag_size_m,
        camera.calibration_matrix.tobytes(),
        camera.distortion_coefficients.tobytes(),
    )


def render_simulation_corpus(
    profile: PlatformProfile,
    *,
    suite: str = "standard",
    seed: int = 2026,
    supersample: int = 1,
) -> SimulationCorpus:
    specs = simulation_frames(suite)
    tag_layout = synthetic_wall_layout()
    renderer = SyntheticAprilTagRenderer(
        profile.camera,
        profile.tag_size_m,
        tag_layout,
        seed=seed,
        supersample=supersample,
    )
    render_ms: list[float] = []
    rendered: list[RenderedFrame] = []
    for spec in specs:
        start = perf_counter()
        rendered.append(renderer.render(spec))
        render_ms.append((perf_counter() - start) * 1000.0)
    return SimulationCorpus(
        tuple(rendered),
        tag_layout,
        tuple(render_ms),
        suite,
        seed,
        supersample,
        _profile_key(profile),
    )


def run_simulation(
    profile: PlatformProfile,
    *,
    suite: str = "standard",
    seed: int = 2026,
    supersample: int = 1,
    repetitions: int = 3,
    warmup_passes: int = 1,
    detector: OpenCVArucoFiducialDetector | None = None,
    corpus: SimulationCorpus | None = None,
) -> SimulationRun:
    if repetitions < 1 or warmup_passes < 0:
        raise ValueError("repetitions must be positive and warmup_passes must be non-negative")
    if corpus is None:
        corpus = render_simulation_corpus(profile, suite=suite, seed=seed, supersample=supersample)
    elif (
        corpus.suite != suite
        or corpus.seed != seed
        or corpus.supersample != supersample
        or corpus.profile_key != _profile_key(profile)
    ):
        raise ValueError("reused simulation corpus does not match the requested profile, suite, seed, or supersample")
    frames = corpus.frames
    tag_layout = corpus.tag_layout
    service_profile = replace(
        profile,
        tag_layout=tag_layout,
        production=False,
        camera_pose_bounds=CameraPoseBounds(
            field_length_m=5.0,
            field_width_m=3.0,
            field_margin_m=0.25,
            min_camera_height_m=0.0,
            max_camera_height_m=2.5,
        ),
    )
    measured_source = frames * (warmup_passes + repetitions)
    scorer = SimulationScoringPublisher(frames, warmup_passes=warmup_passes)
    service = NorthstarVisionService(
        service_profile,
        scorer,
        frame_source=SyntheticFrameSource(measured_source),
        detector=detector or OpenCVArucoFiducialDetector(),
    )
    service.run()
    summary = {
        "schema_version": 1,
        "evidence_kind": "synthetic",
        "generated_at": datetime.now(UTC).isoformat(),
        "limitations": list(SIMULATION_LIMITATIONS),
        "run": {
            "profile": profile.name,
            "suite": suite,
            "seed": seed,
            "supersample": supersample,
            "frames": len(frames),
            "repetitions": repetitions,
            "warmup_passes": warmup_passes,
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "render_timing_ms": metric_distribution(corpus.render_ms),
        **scorer.summary(),
    }
    return SimulationRun(frames, tag_layout, summary, corpus)


if __name__ == "__main__":
    from .simulation_cli import main

    raise SystemExit(main())
