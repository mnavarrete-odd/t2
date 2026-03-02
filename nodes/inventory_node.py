#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List, Optional

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from pallet_vision.msg import Detection, DetectionArray, PickerEvent

try:
    from cenco_interfaces.msg import CountStatus as CountStatusMsg

    HAS_CENCO_COUNT_STATUS = True
except Exception:
    from pallet_vision.msg import CountResult as CountStatusMsg

    HAS_CENCO_COUNT_STATUS = False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.inventory import (  # noqa: E402
    CameraRuntime,
    CounterAdapter,
    DebugStorage,
    DetectorManager,
    InventoryEventManager,
    PhotographerAdapter,
)


class InventoryNode(Node):
    def __init__(self):
        super().__init__("inventory_node")
        self._declare_parameters()
        self._load_parameters()

        self._transition_lock = threading.Lock()
        self._event_manager = InventoryEventManager()

        self._setup_publishers()
        self._setup_core_components()
        self._setup_subscribers()

        self.current_task_context: Dict[str, object] = {}
        self.last_order_result: Dict[str, object] = {}

        self._state_timer = self.create_timer(1.0, self._publish_state)
        self.get_logger().info(
            f"Inventory ready cameras={self.camera_names} "
            f"detector_mode={self.detector_mode} detection={self.enable_detection}"
        )

    def _declare_parameters(self):
        self.declare_parameter("camera_names", "primary_camera,secondary_camera")

        self.declare_parameter("enable_detection", True)
        self.declare_parameter("detector_mode", "shared")
        self.declare_parameter("detection_fps", 10.0)
        self.declare_parameter("detection_mode", "realtime")
        self.declare_parameter("detector_model_path", "models/11-NEW.pt")
        self.declare_parameter("detector_confidence", 0.3)

        self.declare_parameter("sync_queue_size", 100)
        self.declare_parameter("sync_max_delay", 0.3)
        self.declare_parameter("max_frame_queue_size", 200)

        self.declare_parameter("topic_picker_events", "/picker/events")
        self.declare_parameter("topic_inventory_state", "/inventory/state")
        self.declare_parameter("topic_order_result", "/inventory/order_result")
        self.declare_parameter("topic_count_primary", "/inventory/count/primary")
        self.declare_parameter("topic_count_secondary", "/inventory/count/secondary")
        self.declare_parameter("topic_perf_primary", "/inventory/perf/primary")
        self.declare_parameter("topic_perf_secondary", "/inventory/perf/secondary")
        self.declare_parameter("topic_detections_primary", "/inventory/detections/primary")
        self.declare_parameter("topic_detections_secondary", "/inventory/detections/secondary")

        self.declare_parameter("photographer_enabled", True)
        self.declare_parameter("photographer_clear_events", False)
        self.declare_parameter("photographer_base_dir", "/tmp/pallet_vision")
        self.declare_parameter("photographer_area_classes", "area_de_trabajo_pallet")
        self.declare_parameter("photographer_box_classes", "cajas,folio,manga,saco,producto")
        self.declare_parameter("photographer_person_classes", "persona")
        self.declare_parameter("photographer_hand_classes", "mano")
        self.declare_parameter("photographer_empty_classes", "area_de_trabajo_pallet")

        self.declare_parameter("counter_config_yaml", "config/counter_default.yaml")
        self.declare_parameter("counter_device", "auto")

        self.declare_parameter("boundary_wait_timeout_ms", 1500)
        self.declare_parameter("boundary_max_frame_age_ms", 500)

        self.declare_parameter("debug.save_kfs", False)
        self.declare_parameter("debug.save_only_boundary_kfs", True)
        self.declare_parameter("debug.save_counter_frames", False)
        self.declare_parameter("debug.save_counter_video", False)
        self.declare_parameter("debug.output_root", "/tmp/pallet_vision_debug")
        self.declare_parameter("debug.video_fps", 4)

    def _load_parameters(self):
        camera_names_str = self._get_str("camera_names")
        self.camera_names = [x.strip() for x in camera_names_str.split(",") if x.strip()]
        if not self.camera_names:
            self.camera_names = ["primary_camera", "secondary_camera"]

        self.enable_detection = self._get_bool("enable_detection")
        self.detector_mode = self._get_str("detector_mode")
        self.detection_fps = self._get_float("detection_fps")
        self.detection_mode = self._get_str("detection_mode")
        self.detector_model_path = self._resolve_path(self._get_str("detector_model_path"))
        self.detector_confidence = self._get_float("detector_confidence")

        self.sync_queue_size = self._get_int("sync_queue_size")
        self.sync_max_delay = self._get_float("sync_max_delay")
        self.max_frame_queue_size = self._get_int("max_frame_queue_size")

        self.topic_picker_events = self._get_str("topic_picker_events")
        self.topic_inventory_state = self._get_str("topic_inventory_state")
        self.topic_order_result = self._get_str("topic_order_result")
        self.topic_count_primary = self._get_str("topic_count_primary")
        self.topic_count_secondary = self._get_str("topic_count_secondary")
        self.topic_perf_primary = self._get_str("topic_perf_primary")
        self.topic_perf_secondary = self._get_str("topic_perf_secondary")
        self.topic_detections_primary = self._get_str("topic_detections_primary")
        self.topic_detections_secondary = self._get_str("topic_detections_secondary")

        self.photographer_enabled = self._get_bool("photographer_enabled")
        self.photographer_clear_events = self._get_bool("photographer_clear_events")
        self.photographer_base_dir = self._resolve_path(self._get_str("photographer_base_dir"))
        self.photographer_area_classes = self._csv_to_list(self._get_str("photographer_area_classes"))
        self.photographer_box_classes = self._csv_to_list(self._get_str("photographer_box_classes"))
        self.photographer_person_classes = self._csv_to_list(self._get_str("photographer_person_classes"))
        self.photographer_hand_classes = self._csv_to_list(self._get_str("photographer_hand_classes"))
        self.photographer_empty_classes = self._csv_to_list(self._get_str("photographer_empty_classes"))

        self.counter_config_yaml = self._resolve_path(self._get_str("counter_config_yaml"))
        self.counter_device = self._get_str("counter_device")

        self.boundary_wait_timeout_ms = self._get_int("boundary_wait_timeout_ms")
        self.boundary_max_frame_age_ms = self._get_int("boundary_max_frame_age_ms")

        self.debug_save_kfs = self._get_bool("debug.save_kfs")
        self.debug_save_only_boundary_kfs = self._get_bool("debug.save_only_boundary_kfs")
        self.debug_save_counter_frames = self._get_bool("debug.save_counter_frames")
        self.debug_save_counter_video = self._get_bool("debug.save_counter_video")
        self.debug_output_root = self._resolve_path(self._get_str("debug.output_root"))
        self.debug_video_fps = self._get_int("debug.video_fps")

    def _setup_publishers(self):
        self.inventory_state_pub = self.create_publisher(String, self.topic_inventory_state, 10)
        self.order_result_pub = self.create_publisher(String, self.topic_order_result, 10)

        self.count_publishers: Dict[str, object] = {}
        self.perf_publishers: Dict[str, object] = {}
        self.detection_publishers: Dict[str, object] = {}

        for idx, cam in enumerate(self.camera_names):
            alias = self._camera_alias(cam)
            if idx == 0:
                count_topic = self.topic_count_primary
                perf_topic = self.topic_perf_primary
                det_topic = self.topic_detections_primary
            elif idx == 1:
                count_topic = self.topic_count_secondary
                perf_topic = self.topic_perf_secondary
                det_topic = self.topic_detections_secondary
            else:
                count_topic = f"/inventory/count/{alias}"
                perf_topic = f"/inventory/perf/{alias}"
                det_topic = f"/inventory/detections/{alias}"

            self.count_publishers[cam] = self.create_publisher(CountStatusMsg, count_topic, 10)
            self.perf_publishers[cam] = self.create_publisher(String, perf_topic, 10)
            self.detection_publishers[cam] = self.create_publisher(DetectionArray, det_topic, 10)

    def _setup_core_components(self):
        self.detector_manager: Optional[DetectorManager] = None
        if self.enable_detection:
            if not Path(self.detector_model_path).exists():
                raise FileNotFoundError(
                    "Detector model not found. "
                    f"detector_model_path='{self.detector_model_path}'. "
                    "Set an absolute path in config/inventory_node_config.yaml "
                    "or place the model under share/pallet_vision/models."
                )
            self.detector_manager = DetectorManager(
                model_path=self.detector_model_path,
                confidence=self.detector_confidence,
                mode=self.detector_mode,
                camera_names=tuple(self.camera_names),
            )

        self.debug_storage = DebugStorage(
            output_root=self.debug_output_root,
            save_kfs=self.debug_save_kfs,
            save_only_boundary_kfs=self.debug_save_only_boundary_kfs,
            save_counter_frames=self.debug_save_counter_frames,
            save_counter_video=self.debug_save_counter_video,
            video_fps=self.debug_video_fps,
        )

        self.photographers: Dict[str, PhotographerAdapter] = {}
        self.counters: Dict[str, CounterAdapter] = {}
        self.runtimes: Dict[str, CameraRuntime] = {}

        for cam in self.camera_names:
            out_dir = str(Path(self.photographer_base_dir) / cam)
            self.photographers[cam] = PhotographerAdapter(
                camera_name=cam,
                out_dir=out_dir,
                area_classes=self.photographer_area_classes,
                box_classes=self.photographer_box_classes,
                person_classes=self.photographer_person_classes,
                hand_classes=self.photographer_hand_classes,
                empty_classes=self.photographer_empty_classes,
                enabled=False,
                clear_events=self.photographer_clear_events,
            )
            self.counters[cam] = CounterAdapter(
                config_path=self.counter_config_yaml,
                device=self.counter_device,
            )
            self.runtimes[cam] = CameraRuntime(
                node=self,
                camera_name=cam,
                process_callback=self._process_camera_frame,
                sync_queue_size=self.sync_queue_size,
                sync_max_delay=self.sync_max_delay,
                max_queue_size=self.max_frame_queue_size,
                detection_fps=self.detection_fps,
                detection_mode=self.detection_mode,
            )
            self.runtimes[cam].set_processing_enabled(False)

    def _setup_subscribers(self):
        self.event_sub = self.create_subscription(
            PickerEvent,
            self.topic_picker_events,
            self._on_picker_event,
            20,
        )

    def _on_picker_event(self, msg: PickerEvent):
        payload = {
            "event_type": msg.event_type,
            "order_id": msg.order_id,
            "hu_id": msg.hu_id,
            "sku": msg.sku,
            "description": msg.description,
            "quantity": msg.quantity,
            "item_index": msg.item_index,
            "hu_index": msg.hu_index,
            "total_hus": msg.total_hus,
        }

        decision = self._event_manager.process_event(payload)
        if decision.ignored:
            self.get_logger().warn(f"Ignored event {msg.event_type}: {decision.reason}")
            self._publish_state()
            return

        with self._transition_lock:
            if decision.end_task:
                self._end_task()

            if decision.close_previous_order and decision.previous_order_id:
                self._finalize_order(order_id=decision.previous_order_id, cause="order_switch")

            if decision.start_task:
                self.current_task_context = self._event_manager.task_context
                self._start_task(self.current_task_context)

            if decision.close_order:
                self._finalize_order(order_id=msg.order_id, cause="order_end")

        self._publish_state()

    def _start_task(self, task_context: Dict[str, object]):
        order_id = str(task_context.get("order_id", "ORDER"))
        hu_id = str(task_context.get("hu_id", "HU"))
        sku = str(task_context.get("sku", "SKU"))
        task_id = f"{order_id}_{hu_id}_{sku}".replace("/", "_")

        self.debug_storage.begin_task(task_id, self.camera_names)

        for cam in self.camera_names:
            request = self._build_boundary_request(cam, event_type="KF-TASK-START")
            if request is not None:
                self._consume_keyframe(cam, request, is_boundary=True)

        for cam in self.camera_names:
            self.photographers[cam].set_enabled(self.photographer_enabled)
            self.runtimes[cam].set_processing_enabled(True)

        self.get_logger().info(f"Task started order={order_id} hu={hu_id} sku={sku}")

    def _end_task(self):
        for cam in self.camera_names:
            request = self._build_boundary_request(cam, event_type="KF-TASK-END")
            if request is not None:
                self._consume_keyframe(cam, request, is_boundary=True)

        for cam in self.camera_names:
            self.runtimes[cam].set_processing_enabled(False)
            self.photographers[cam].set_enabled(False)

        self.debug_storage.end_task()
        self.current_task_context = {}
        self.get_logger().info("Task ended")

    def _finalize_order(self, order_id: str, cause: str):
        cameras = {}
        for cam in self.camera_names:
            snapshot = self.counters[cam].get_snapshot()
            cameras[self._camera_alias(cam)] = {
                "running_units": snapshot.running_units,
                "added_units": snapshot.added_units,
                "removed_units": snapshot.removed_units,
                "net_units": snapshot.net_units,
                "num_active_tracks": snapshot.num_active_tracks,
                "processed_keyframes": self.counters[cam].processed_keyframes,
            }

        payload = {
            "order_id": order_id,
            "cause": cause,
            "timestamp": time.time(),
            "cameras": cameras,
        }
        self.last_order_result = payload

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.order_result_pub.publish(msg)

        for cam in self.camera_names:
            self.runtimes[cam].set_processing_enabled(False)
            self.photographers[cam].set_enabled(False)
            self.counters[cam].reset()

        self.debug_storage.end_task()
        self.current_task_context = {}
        self.get_logger().info(f"Order finalized order_id={order_id} cause={cause}")

    def _process_camera_frame(self, frame):
        cam = frame.camera_name
        if not self._event_manager.task_active:
            return

        t0 = time.time()
        detections, photo_dets = self._run_detection(cam, frame.rgb_image)
        self._publish_detection_array(cam, frame, detections)

        output = self.photographers[cam].process_frame(
            frame_id=frame.frame_id,
            rgb_image=frame.rgb_image,
            depth_image=frame.depth_image,
            photographer_detections=photo_dets,
        )
        for request in output.save_requests:
            self._consume_keyframe(cam, request, is_boundary=False)

        elapsed_ms = (time.time() - t0) * 1000.0
        perf_payload = {
            "camera_name": cam,
            "frame_id": frame.frame_id,
            "detections": len(detections),
            "keyframes": len(output.save_requests),
            "elapsed_ms": elapsed_ms,
            "runtime": self.runtimes[cam].get_stats(),
        }
        self._publish_perf(cam, perf_payload)

    def _build_boundary_request(self, cam: str, event_type: str):
        frame = self._wait_latest_frame(cam)
        if frame is None:
            self.get_logger().warn(f"[{cam}] no frame available for {event_type}")
            return None

        _, photo_dets = self._run_detection(cam, frame.rgb_image)
        return self.photographers[cam].force_boundary_keyframe(
            event_type=event_type,
            frame_id=frame.frame_id,
            rgb_image=frame.rgb_image,
            depth_image=frame.depth_image,
            photographer_detections=photo_dets,
        )

    def _run_detection(self, cam: str, image):
        if not self.enable_detection or self.detector_manager is None:
            return [], []
        result = self.detector_manager.detect(cam, image)
        return result.detections, result.photographer_detections

    def _consume_keyframe(self, cam: str, request, is_boundary: bool):
        snapshot = self.counters[cam].process_keyframe(request)
        self.debug_storage.save_keyframe(camera_name=cam, request=request, is_boundary=is_boundary)

        if self.debug_storage.save_counter_frames:
            frame_data, prepared, result = self.counters[cam].get_last_debug_bundle()
            if frame_data is not None and prepared is not None and result is not None:
                self.debug_storage.save_counter_frame(
                    camera_name=cam,
                    frame_data=frame_data,
                    prepared=prepared,
                    result=result,
                )

        self._publish_count_status(cam, snapshot)

    def _wait_latest_frame(self, cam: str):
        runtime = self.runtimes[cam]
        deadline = time.time() + (self.boundary_wait_timeout_ms / 1000.0)
        while time.time() < deadline:
            frame = runtime.get_latest_frame(self.boundary_max_frame_age_ms)
            if frame is not None:
                return frame
            time.sleep(0.01)
        return None

    def _publish_count_status(self, cam: str, snapshot):
        msg = CountStatusMsg()
        if hasattr(msg, "header"):
            msg.header.stamp = self.get_clock().now().to_msg()
        if hasattr(msg, "stamp"):
            msg.stamp = self.get_clock().now().to_msg()
        if hasattr(msg, "camera_name"):
            msg.camera_name = self._camera_alias(cam)
        if hasattr(msg, "slot_name"):
            msg.slot_name = self._camera_alias(cam)
        if hasattr(msg, "running_units"):
            msg.running_units = float(snapshot.running_units)
        if hasattr(msg, "net_units"):
            msg.net_units = float(snapshot.net_units)
        if hasattr(msg, "added_units"):
            msg.added_units = float(snapshot.added_units)
        if hasattr(msg, "removed_units"):
            msg.removed_units = float(snapshot.removed_units)
        if hasattr(msg, "change_state"):
            msg.change_state = snapshot.change_state
        if hasattr(msg, "change_detail"):
            msg.change_detail = snapshot.change_detail
        if hasattr(msg, "num_active_tracks"):
            msg.num_active_tracks = int(snapshot.num_active_tracks)
        if hasattr(msg, "frame_index"):
            msg.frame_index = int(snapshot.frame_index)
        if not HAS_CENCO_COUNT_STATUS:
            if hasattr(msg, "num_detections"):
                msg.num_detections = int(snapshot.num_detections)
            if hasattr(msg, "num_matched"):
                msg.num_matched = int(snapshot.num_matched)
            if hasattr(msg, "num_new"):
                msg.num_new = int(snapshot.num_new)
            if hasattr(msg, "num_lost_tracks"):
                msg.num_lost_tracks = int(snapshot.num_lost_tracks)

        self.count_publishers[cam].publish(msg)

    def _publish_detection_array(self, cam: str, frame, detections: List[dict]):
        msg = DetectionArray()
        msg.stamp = frame.rgb_msg.header.stamp
        msg.camera_name = self._camera_alias(cam)
        msg.frame_id = frame.frame_id

        ros_dets = []
        for d in detections:
            dm = Detection()
            dm.class_id = int(d["class_id"])
            dm.class_name = str(d["class_name"])
            dm.confidence = float(d["confidence"])
            x1, y1, x2, y2 = d["bbox"]
            dm.bbox_x1 = int(x1)
            dm.bbox_y1 = int(y1)
            dm.bbox_x2 = int(x2)
            dm.bbox_y2 = int(y2)
            ros_dets.append(dm)

        msg.detections = ros_dets
        self.detection_publishers[cam].publish(msg)

    def _publish_perf(self, cam: str, payload: Dict[str, object]):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.perf_publishers[cam].publish(msg)

    def _publish_state(self):
        state = self._event_manager.get_state()
        state["camera_stats"] = {
            self._camera_alias(cam): self.runtimes[cam].get_stats()
            for cam in self.camera_names
        }
        state["counts"] = {
            self._camera_alias(cam): self.counters[cam].get_snapshot().__dict__
            for cam in self.camera_names
        }
        if self.last_order_result:
            state["last_order_result"] = self.last_order_result

        msg = String()
        msg.data = json.dumps(state, ensure_ascii=True)
        self.inventory_state_pub.publish(msg)

    def _camera_alias(self, camera_name: str) -> str:
        out = camera_name.strip()
        if out.endswith("_camera"):
            out = out[: -len("_camera")]
        return out or camera_name

    def _csv_to_list(self, raw: str) -> List[str]:
        return [x.strip() for x in (raw or "").split(",") if x.strip()]

    def _resolve_path(self, raw: str) -> str:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return str(p)

        # 1) Package source path (development mode).
        source_path = (PROJECT_ROOT / p).resolve()
        if source_path.exists():
            return str(source_path)

        # 2) Installed package share path (runtime mode).
        share_candidate = None
        try:
            pkg_share = Path(get_package_share_directory("pallet_vision"))
            share_candidate = (pkg_share / p).resolve()
            if share_candidate.exists():
                return str(share_candidate)
        except Exception:
            share_candidate = None

        # 3) Current working directory as fallback.
        cwd_path = (Path(os.getcwd()) / p).resolve()
        if cwd_path.exists():
            return str(cwd_path)

        # Return best-effort candidate so error messages show expected location.
        if share_candidate is not None:
            return str(share_candidate)
        return str(source_path)

    def _get_str(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _get_bool(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _get_int(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _get_float(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def cleanup(self):
        for runtime in self.runtimes.values():
            runtime.stop()


def main(args=None):
    rclpy.init(args=args)
    node = InventoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
