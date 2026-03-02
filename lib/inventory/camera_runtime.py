from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock, Thread
import time
from typing import Callable, Optional

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


@dataclass
class SyncedFrame:
    camera_name: str
    frame_id: int
    received_time: float
    rgb_msg: Image
    depth_msg: Image
    rgb_image: object
    depth_image: object


class CameraRuntime:
    def __init__(
        self,
        node: Node,
        camera_name: str,
        process_callback: Callable[[SyncedFrame], None],
        *,
        sync_queue_size: int = 100,
        sync_max_delay: float = 0.3,
        max_queue_size: int = 200,
        detection_fps: float = 0.0,
        detection_mode: str = "realtime",
    ):
        self.node = node
        self.camera_name = camera_name
        self.process_callback = process_callback
        self.max_queue_size = max(1, int(max_queue_size))
        self.detection_mode = (detection_mode or "realtime").strip().lower()
        if self.detection_mode not in {"realtime", "no_drop"}:
            self.detection_mode = "realtime"

        self.detection_period = 1.0 / float(detection_fps) if detection_fps > 0 else 0.0
        self.last_enqueue_time = 0.0
        self.processing_enabled = False

        self.bridge = CvBridge()
        self._latest_lock = Lock()
        self._queue_lock = Lock()
        self._latest_frame: Optional[SyncedFrame] = None
        self._queue: deque[SyncedFrame] = deque()
        self._running = True
        self._worker = Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        qos = qos_profile_sensor_data
        self.rgb_sub = Subscriber(
            node, Image, f"/{camera_name}/color/image_raw", qos_profile=qos
        )
        self.depth_sub = Subscriber(
            node, Image, f"/{camera_name}/depth/image_raw", qos_profile=qos
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            sync_queue_size,
            sync_max_delay,
        )
        self.sync.registerCallback(self._on_sync)

        self.frames_received = 0
        self.frames_enqueued = 0
        self.frames_processed = 0
        self.frames_dropped = 0

    def set_processing_enabled(self, enabled: bool) -> None:
        self.processing_enabled = bool(enabled)
        if not enabled:
            with self._queue_lock:
                self._queue.clear()

    def get_latest_frame(self, max_age_ms: int) -> Optional[SyncedFrame]:
        now = time.time()
        with self._latest_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        age_ms = (now - frame.received_time) * 1000.0
        if age_ms > float(max_age_ms):
            return None
        return frame

    def get_stats(self) -> dict:
        with self._queue_lock:
            qsize = len(self._queue)
        return {
            "camera_name": self.camera_name,
            "frames_received": self.frames_received,
            "frames_enqueued": self.frames_enqueued,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "queue_size": qsize,
            "processing_enabled": self.processing_enabled,
        }

    def stop(self) -> None:
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        with self._queue_lock:
            self._queue.clear()
        with self._latest_lock:
            self._latest_frame = None

    def _on_sync(self, rgb_msg: Image, depth_msg: Image) -> None:
        now = time.time()
        self.frames_received += 1

        try:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:
            self.node.get_logger().warning(
                f"[{self.camera_name}] cv_bridge conversion failed: {exc}"
            )
            return

        frame = SyncedFrame(
            camera_name=self.camera_name,
            frame_id=self.frames_received,
            received_time=now,
            rgb_msg=rgb_msg,
            depth_msg=depth_msg,
            rgb_image=rgb_img,
            depth_image=depth_img,
        )

        with self._latest_lock:
            self._latest_frame = frame

        if not self.processing_enabled:
            return

        if self.detection_period > 0:
            if now - self.last_enqueue_time < self.detection_period:
                return
            self.last_enqueue_time = now

        with self._queue_lock:
            if self.detection_mode == "realtime":
                if len(self._queue) > 0:
                    self.frames_dropped += len(self._queue)
                    self._queue.clear()
                self._queue.append(frame)
            else:
                if len(self._queue) >= self.max_queue_size:
                    self._queue.popleft()
                    self.frames_dropped += 1
                self._queue.append(frame)
            self.frames_enqueued += 1

    def _worker_loop(self) -> None:
        while self._running:
            frame = None
            with self._queue_lock:
                if self._queue:
                    frame = self._queue.popleft()
            if frame is None:
                time.sleep(0.002)
                continue

            try:
                self.process_callback(frame)
                self.frames_processed += 1
            except Exception as exc:
                self.node.get_logger().error(
                    f"[{self.camera_name}] processing error: {exc}"
                )
