import numpy as np

from core.photographer.types import DetectionData
from lib.inventory.photographer_adapter import PhotographerAdapter


def test_force_boundary_keyframe_contains_depth_and_event_type(tmp_path):
    adapter = PhotographerAdapter(
        camera_name="primary_camera",
        out_dir=str(tmp_path / "primary"),
        area_classes=["area_de_trabajo_pallet"],
        box_classes=["producto"],
        person_classes=["persona"],
        hand_classes=["mano"],
        empty_classes=["area_de_trabajo_pallet"],
        enabled=True,
        clear_events=False,
    )

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.float32)
    dets = [
        DetectionData(
            class_id=1,
            class_name="producto",
            bbox=(10.0, 10.0, 20.0, 20.0),
            confidence=0.9,
        )
    ]

    request = adapter.force_boundary_keyframe(
        event_type="KF-TASK-START",
        frame_id=7,
        rgb_image=rgb,
        depth_image=depth,
        photographer_detections=dets,
    )

    assert request.event_type == "KF-TASK-START"
    assert request.frame_index == 7
    assert request.original_depth is depth
    assert len(request.detections) == 1
