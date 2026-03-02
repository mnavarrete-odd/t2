from .event_manager import EventDecision, InventoryEventManager
from .counter_adapter import CounterAdapter, CounterSnapshot
from .debug_storage import DebugStorage
from .photographer_adapter import PhotographerAdapter, PhotographerOutput

__all__ = [
    "EventDecision",
    "InventoryEventManager",
    "CounterAdapter",
    "CounterSnapshot",
    "DebugStorage",
    "PhotographerAdapter",
    "PhotographerOutput",
    "CameraRuntime",
    "SyncedFrame",
    "DetectionResult",
    "DetectorManager",
]


def __getattr__(name: str):
    if name in {"CameraRuntime", "SyncedFrame"}:
        from .camera_runtime import CameraRuntime, SyncedFrame

        return {"CameraRuntime": CameraRuntime, "SyncedFrame": SyncedFrame}[name]
    if name in {"DetectionResult", "DetectorManager"}:
        from .detector_manager import DetectionResult, DetectorManager

        return {"DetectionResult": DetectionResult, "DetectorManager": DetectorManager}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
