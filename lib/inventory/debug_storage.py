from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from core.photographer.types import KeyframeSaveRequest
from lib.debug_writer.debug_writer import DebugWriter


class DebugStorage:
    def __init__(
        self,
        *,
        output_root: str,
        save_kfs: bool,
        save_only_boundary_kfs: bool,
        save_counter_frames: bool,
        save_counter_video: bool,
        video_fps: int = 4,
    ):
        self.output_root = Path(output_root)
        self.save_kfs = bool(save_kfs)
        self.save_only_boundary_kfs = bool(save_only_boundary_kfs)
        self.save_counter_frames = bool(save_counter_frames)
        self.save_counter_video = bool(save_counter_video)
        self.video_fps = int(video_fps)
        self._writers: Dict[str, DebugWriter] = {}
        self._active_task_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.save_kfs or self.save_counter_frames or self.save_counter_video

    def begin_task(self, task_id: str, camera_names: list[str]) -> None:
        self._active_task_id = task_id
        if not self.enabled:
            return
        self._writers = {}
        for camera_name in camera_names:
            writer = DebugWriter(
                output_dir=str(self.output_root / camera_name),
                save_keyframes=self.save_kfs,
                save_counter_frames=self.save_counter_frames,
                save_counter_video=self.save_counter_video,
                video_fps=self.video_fps,
            )
            writer.begin_task(task_id)
            self._writers[camera_name] = writer

    def save_keyframe(
        self,
        *,
        camera_name: str,
        request: KeyframeSaveRequest,
        is_boundary: bool,
    ) -> None:
        if not self.enabled or not self.save_kfs:
            return
        if self.save_only_boundary_kfs and not is_boundary:
            return
        writer = self._writers.get(camera_name)
        if writer is None:
            return
        writer.save_keyframe(request)

    def save_counter_frame(self, *, camera_name: str, frame_data, prepared, result) -> None:
        if not self.enabled or not self.save_counter_frames:
            return
        writer = self._writers.get(camera_name)
        if writer is None:
            return
        writer.save_tracking_frame(frame_data, prepared, result)

    def end_task(self) -> None:
        if not self.enabled:
            self._active_task_id = None
            return
        for writer in self._writers.values():
            writer.end_task()
        self._writers = {}
        self._active_task_id = None
