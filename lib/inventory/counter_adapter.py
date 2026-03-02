from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.counter.types import FrameResult
from core.photographer.types import KeyframeSaveRequest


@dataclass
class CounterSnapshot:
    running_units: float
    net_units: float
    added_units: float
    removed_units: float
    change_state: str
    change_detail: str
    num_active_tracks: int
    frame_index: int
    num_detections: int
    num_matched: int
    num_new: int
    num_lost_tracks: int


class CounterAdapter:
    def __init__(self, config_path: str, device: str = "auto", bridge=None):
        if bridge is None:
            from lib.counter_bridge.counter_bridge import CounterBridge

            bridge = CounterBridge(config_path=config_path, device=device)
        self.bridge = bridge
        self.last_result: Optional[FrameResult] = None
        self.processed_keyframes = 0

    def reset(self) -> None:
        self.bridge.reset()
        self.last_result = None
        self.processed_keyframes = 0

    def process_keyframe(self, request: KeyframeSaveRequest) -> CounterSnapshot:
        result = self.bridge.process_keyframe(
            request, depth_map=request.original_depth
        )
        self.last_result = result
        self.processed_keyframes += 1
        return self._to_snapshot(result)

    def get_last_debug_bundle(self):
        return (
            self.bridge.last_frame_data,
            self.bridge.last_prepared,
            self.bridge.last_result,
        )

    def get_snapshot(self) -> CounterSnapshot:
        if self.last_result is None:
            return CounterSnapshot(
                running_units=0.0,
                net_units=0.0,
                added_units=0.0,
                removed_units=0.0,
                change_state="no_change",
                change_detail="none",
                num_active_tracks=0,
                frame_index=0,
                num_detections=0,
                num_matched=0,
                num_new=0,
                num_lost_tracks=0,
            )
        return self._to_snapshot(self.last_result)

    @staticmethod
    def _to_snapshot(result: FrameResult) -> CounterSnapshot:
        return CounterSnapshot(
            running_units=float(result.running_units),
            net_units=float(result.net_units),
            added_units=float(result.added_units),
            removed_units=float(result.removed_units),
            change_state=str(result.change_state),
            change_detail=str(result.change_detail),
            num_active_tracks=int(result.num_active_tracks),
            frame_index=int(result.frame_index),
            num_detections=int(result.num_detections),
            num_matched=int(result.num_matched),
            num_new=int(result.num_new),
            num_lost_tracks=int(result.num_lost_tracks),
        )
