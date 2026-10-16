from __future__ import annotations

from collections import deque
from time import perf_counter

import cv2

from .capture import FrameSource, make_frame_source
from .config import PlatformProfile
from .fiducials import OpenCVArucoFiducialDetector
from .orientation import RobotYawSource, StaticRobotYawSource
from .output import OutputPublisher
from .overlay import draw_overlay
from .prefilter import BlockSkipPrefilter
from .solvers import CameraPoseSolver, SingleTagPoseSolver, TagAngleSolver, TrigDistanceSolver
from .types import FrameResult


class NorthstarVisionService:
    def __init__(
        self,
        profile: PlatformProfile,
        output: OutputPublisher,
        *,
        replay_path: str | None = None,
        loop_replay: bool = False,
        show: bool = False,
        robot_yaw_rad: float | None = None,
        yaw_source: RobotYawSource | None = None,
        frame_source: FrameSource | None = None,
        detector: OpenCVArucoFiducialDetector | None = None,
    ) -> None:
        self.profile = profile
        self.output = output
        self.show = show
        if robot_yaw_rad is not None and yaw_source is not None:
            raise ValueError("Provide either robot_yaw_rad or yaw_source, not both")
        self.yaw_source = yaw_source
        if self.yaw_source is None and robot_yaw_rad is not None:
            self.yaw_source = StaticRobotYawSource(robot_yaw_rad)
        prefilter = BlockSkipPrefilter(profile.skip_scan) if profile.skip_scan.enabled else None
        self.detector = detector or OpenCVArucoFiducialDetector(prefilter=prefilter)
        self.single_solver = SingleTagPoseSolver(profile.max_reprojection_error_px)
        self.camera_solver = CameraPoseSolver(
            profile.max_reprojection_error_px,
            max_tag_reprojection_error_px=profile.max_tag_reprojection_error_px,
            bounds=profile.camera_pose_bounds,
            max_outlier_iterations=profile.max_outlier_iterations,
            ransac_iterations=profile.ransac_iterations,
        )
        self.angle_solver = TagAngleSolver()
        self.trig_solver = TrigDistanceSolver()
        self.frame_source = frame_source or make_frame_source(profile, replay_path, loop_replay=loop_replay)
        self.frame_times: deque[float] = deque(maxlen=60)
        self.previous_publish_ms = 0.0
        self.closed = False

    def process_next(self) -> bool:
        frame_start = perf_counter()
        ok, timestamp, capture_monotonic_s, frame = self.frame_source.read()
        capture_end = perf_counter()
        if not ok or frame is None:
            if self.frame_source.is_live:
                raise RuntimeError("Live camera stopped returning frames")
            return False
        actual_height, actual_width = frame.shape[:2]
        expected = self.profile.camera
        if actual_width != expected.width or actual_height != expected.height:
            raise RuntimeError(
                f"Frame is {actual_width}x{actual_height}, but profile calibration requires "
                f"{expected.width}x{expected.height}"
            )

        detection_start = perf_counter()
        observations = self.detector.detect(frame)
        detection_end = perf_counter()
        yaw_estimate = None if self.yaw_source is None else self.yaw_source.yaw_at(capture_monotonic_s)
        single_poses = []
        solved_observations = []
        for observation in observations:
            single = self.single_solver.solve(observation, self.profile.camera, self.profile.tag_size_m)
            if single is None:
                continue
            single_poses.append(single)
            solved_observations.append(observation)
        tag_angles = self.angle_solver.solve_many(solved_observations, self.profile.camera, single_poses)
        trig_poses = []
        if yaw_estimate is not None:
            field_rotation = self.trig_solver.yaw_rotation(yaw_estimate.yaw_rad)
            for tag_angle in tag_angles:
                trig = self.trig_solver.solve(
                    tag_angle,
                    self.profile.camera,
                    self.profile.tag_layout,
                    yaw_estimate.yaw_rad,
                    field_rotation=field_rotation,
                )
                if trig is not None:
                    trig_poses.append(trig)
        camera_pose = self.camera_solver.solve(
            observations,
            single_poses,
            self.profile.camera,
            self.profile.tag_size_m,
            self.profile.tag_layout,
        )
        solve_end = perf_counter()
        self.frame_times.append(solve_end)
        processing_ms = (solve_end - capture_end) * 1000.0
        frame_age_ms = max(0.0, (solve_end - capture_monotonic_s) * 1000.0)
        source_health = self._source_health()
        yaw_health = {} if self.yaw_source is None else self.yaw_source.health()
        output_health = self.output.telemetry()
        pose_health = self._pose_health()
        result = FrameResult(
            timestamp=timestamp,
            capture_monotonic_s=capture_monotonic_s,
            image_observations=observations,
            single_tag_poses=single_poses,
            camera_pose=camera_pose,
            pose_diagnostics=self.camera_solver.last_diagnostics,
            tag_angles=tag_angles,
            trig_poses=trig_poses,
            health={
                "fps": self._fps(),
                "latency_ms": frame_age_ms,
                "frame_age_ms": frame_age_ms,
                "acquire_wait_ms": (capture_end - frame_start) * 1000.0,
                "processing_ms": processing_ms,
                "detection_ms": (detection_end - detection_start) * 1000.0,
                "solve_ms": (solve_end - detection_end) * 1000.0,
                "previous_publish_ms": self.previous_publish_ms,
                "source_connected": True,
                "profile": self.profile.name,
                "platform": self.profile.platform,
                "yaw_available": yaw_estimate is not None,
                "yaw_sample_age_ms": None if yaw_estimate is None else yaw_estimate.sample_age_s * 1000.0,
                "yaw_interpolated": False if yaw_estimate is None else yaw_estimate.interpolated,
                "output": output_health,
                "pose_solver": pose_health,
                **source_health,
                **yaw_health,
            },
        )
        publish_start = perf_counter()
        self.output.publish(result)
        self.previous_publish_ms = (perf_counter() - publish_start) * 1000.0
        if self.show:
            cv2.imshow("Northstar Vision", draw_overlay(frame, result))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                return False
        return True

    def run(self, max_frames: int | None = None) -> None:
        try:
            if max_frames is not None and max_frames < 0:
                raise ValueError("max_frames must not be negative")
            frames = 0
            while max_frames is None or frames < max_frames:
                if not self.process_next():
                    break
                frames += 1
        finally:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.frame_source.release()
        finally:
            try:
                if self.yaw_source is not None:
                    self.yaw_source.close()
            finally:
                try:
                    self.output.close()
                finally:
                    if self.show:
                        cv2.destroyAllWindows()

    def _fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return 0.0 if elapsed <= 0 else (len(self.frame_times) - 1) / elapsed

    def _source_health(self) -> dict[str, object]:
        health = getattr(self.frame_source, "health", None)
        return {} if health is None else dict(health())

    def _pose_health(self) -> dict[str, object]:
        diagnostics = self.camera_solver.last_diagnostics
        return {
            "attempted_tag_ids": diagnostics.attempted_tag_ids,
            "accepted_tag_ids": diagnostics.accepted_tag_ids,
            "rejected_tag_ids": sorted(diagnostics.rejected_tag_reasons),
            "overall_rms_reprojection_error_px": diagnostics.overall_rms_reprojection_error_px,
            "used_ransac": diagnostics.used_ransac,
            "rejection_reason": diagnostics.rejection_reason,
        }
