from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

from core.detector_yolo import YOLODetector
from core.photographer.types import DetectionData


@dataclass
class DetectionResult:
    detections: List[dict]
    photographer_detections: List[DetectionData]


class DetectorManager:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        mode: str = "shared",
        camera_names: Tuple[str, ...] = (),
    ):
        self.mode = (mode or "shared").strip().lower()
        if self.mode not in {"shared", "per_camera"}:
            self.mode = "shared"

        self._model_str = str(Path(model_path))
        self._confidence = float(confidence)
        self._shared_lock = Lock()
        self._shared_detector = None
        self._camera_detectors: Dict[str, YOLODetector] = {}

        if self.mode == "shared":
            self._shared_detector = YOLODetector(
                model_path=self._model_str, conf=self._confidence
            )
        else:
            for name in camera_names:
                self._camera_detectors[name] = YOLODetector(
                    model_path=self._model_str, conf=self._confidence
                )

    def detect(self, camera_name: str, image) -> DetectionResult:
        detector = self._get_detector(camera_name)
        if detector is None or image is None:
            return DetectionResult(detections=[], photographer_detections=[])

        if self.mode == "shared":
            with self._shared_lock:
                results, _ = detector.detect(image)
        else:
            results, _ = detector.detect(image)

        detections = self._convert_results(results)
        photo_dets = [
            DetectionData(
                class_id=int(d["class_id"]),
                class_name=str(d["class_name"]),
                bbox=tuple(float(x) for x in d["bbox"]),
                confidence=float(d["confidence"]),
                tracking_id=d.get("tracking_id"),
            )
            for d in detections
        ]
        return DetectionResult(detections=detections, photographer_detections=photo_dets)

    def _get_detector(self, camera_name: str):
        if self.mode == "shared":
            return self._shared_detector
        if camera_name not in self._camera_detectors:
            self._camera_detectors[camera_name] = YOLODetector(
                model_path=self._model_str, conf=self._confidence
            )
        return self._camera_detectors[camera_name]

    @staticmethod
    def _convert_results(results) -> List[dict]:
        out: List[dict] = []
        if not results:
            return out
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out

        names = getattr(result, "names", {})
        track_ids = None
        if getattr(boxes, "id", None) is not None:
            try:
                track_ids = boxes.id.cpu().numpy().astype(int).tolist()
            except Exception:
                track_ids = None

        for idx, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if isinstance(names, dict):
                class_name = names.get(cls, str(cls))
            else:
                class_name = names[cls] if cls < len(names) else str(cls)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            tracking_id = None
            if track_ids is not None and idx < len(track_ids):
                tracking_id = int(track_ids[idx])

            out.append(
                {
                    "class_id": cls,
                    "class_name": str(class_name),
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "tracking_id": tracking_id,
                }
            )
        return out
