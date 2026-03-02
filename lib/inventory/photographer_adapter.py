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
        product_kf_model: str | None = "models/onlyProduct.pt",
        product_kf_conf: float = 0.1,
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
            product_kf_model=Path(product_kf_model) if product_kf_model else None,
            product_kf_conf=float(product_kf_conf),
            area_classes=tuple(area_classes),
            box_classes=tuple(box_classes),
            person_classes=tuple(person_classes),
            hand_classes=tuple(hand_classes),
            empty_classes=tuple(empty_classes),
            capture_all=False,
            clear_events=bool(clear_events),
            person_dist_px=450.0,
            person_near_enabled=False,
            hand_dist_px=400.0,
            product_start_frames=1,
            product_end_frames=1,
            product_pre_offset_frames=0,
            product_post_offset_frames=0,
            product_save_prepost=False,
            area_min_conf=0.1,
            area_warmup_frames=10,
            area_stable_frames=1,
            area_hold_frames=50,
            area_expand_ratio=0.1,
            area_refit_enabled=True,
            area_refit_frames=20,
            area_refit_center_dist_min_px=20.0,
            area_refit_center_dist_max_px=100.0,
            stable_area_frames=5,
            stable_area_movement_max=0.01,
            stable_area_require_count_stability=False,
            movement_by_area=True,
            stable_reconfirm_frames=2,
            stable_empty_frames=5,
            occlusion_start_ratio=0.05,
            occlusion_end_ratio=0.05,
            occlusion_start_frames=1,
            occlusion_end_frames=1,
            occlusion_pre_offset_frames=0,
            occlusion_post_offset_frames=0,
            occlusion_item_classes=(
                "producto",
                "cajas",
                "folio",
                "manga",
                "saco",
                "producto_en_mano",
            ),
            occlusion_change_count_min=1,
            occlusion_change_coverage_min=0.05,
            occlusion_change_confirm_frames=1,
            occlusion_change_cooldown_frames=10,
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
