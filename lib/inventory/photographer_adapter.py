from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from core.photographer.config import PhotographerConfig
from core.photographer.photographer import Photographer
from core.photographer.types import DetectionData, FrameMetrics, KeyframeSaveRequest


@dataclass
class PhotographerOutput:
    detections: List[dict]
    save_requests: List[KeyframeSaveRequest]


class PhotographerAdapter:
    def __init__(
        self,
        *,
        camera_name: str,
        out_dir: str,
        area_classes: List[str],
        box_classes: List[str],
        person_classes: List[str],
        hand_classes: List[str],
        empty_classes: List[str],
        enabled: bool = True,
        clear_events: bool = False,
    ):
        cfg = PhotographerConfig(
            outdir=Path(out_dir),
            depth_dir=None,
            product_kf_model=None,
            area_classes=tuple(area_classes),
            box_classes=tuple(box_classes),
            person_classes=tuple(person_classes),
            hand_classes=tuple(hand_classes),
            empty_classes=tuple(empty_classes),
            clear_events=bool(clear_events),
        )
        self.camera_name = camera_name
        self.photographer = Photographer(cfg)
        self.enabled = bool(enabled)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def process_frame(
        self,
        *,
        frame_id: int,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        photographer_detections: List[DetectionData],
    ) -> PhotographerOutput:
        if not self.enabled:
            return PhotographerOutput(detections=[], save_requests=[])

        _, _, save_requests = self.photographer.update(
            frame_index=frame_id,
            image=rgb_image,
            detections=photographer_detections,
            depth_image=depth_image,
            include_signals=False,
        )
        detections_dict = self._to_detection_dicts(photographer_detections)
        return PhotographerOutput(detections=detections_dict, save_requests=save_requests)

    def force_boundary_keyframe(
        self,
        *,
        event_type: str,
        frame_id: int,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        photographer_detections: List[DetectionData],
    ) -> KeyframeSaveRequest:
        metrics = self._build_min_metrics(
            frame_id=frame_id,
            image_shape=rgb_image.shape,
            count_in_area=len(photographer_detections),
        )
        return KeyframeSaveRequest(
            event_type=event_type,
            frame_index=frame_id,
            image=rgb_image,
            metrics=metrics,
            detections=list(photographer_detections),
            filename_tag=event_type,
            folder_tag=event_type,
            original_image=rgb_image,
            original_depth=depth_image,
        )

    @staticmethod
    def _build_min_metrics(
        *,
        frame_id: int,
        image_shape,
        count_in_area: int,
    ) -> FrameMetrics:
        h, w = image_shape[:2]
        return FrameMetrics(
            frame_index=frame_id,
            image_w=int(w),
            image_h=int(h),
            area_bbox=None,
            area_bbox_raw=None,
            area_class_name=None,
            area_confidence=None,
            area_stable_frames=0,
            count_in_area=int(count_in_area),
            coverage_ratio=0.0,
            movement_score=0.0,
            occlusion_ratio=0.0,
            has_person_near=False,
            class_counts={},
        )

    @staticmethod
    def _to_detection_dicts(dets: List[DetectionData]) -> List[dict]:
        out = []
        for d in dets:
            out.append(
                {
                    "class_id": int(d.class_id),
                    "class_name": str(d.class_name),
                    "confidence": float(d.confidence),
                    "bbox": [float(v) for v in d.bbox],
                }
            )
        return out
