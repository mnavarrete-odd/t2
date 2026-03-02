from types import SimpleNamespace

from lib.inventory.counter_adapter import CounterAdapter


class FakeBridge:
    def __init__(self):
        self.last_frame_data = "frame"
        self.last_prepared = ["prepared"]
        self.last_result = None
        self._running_units = 0.0

    def process_keyframe(self, request, depth_map=None):
        self._running_units += 1.0
        result = SimpleNamespace(
            running_units=self._running_units,
            net_units=1.0,
            added_units=1.0,
            removed_units=0.0,
            change_state="added",
            change_detail="added_only",
            num_active_tracks=1,
            frame_index=getattr(request, "frame_index", 0),
            num_detections=1,
            num_matched=0,
            num_new=1,
            num_lost_tracks=0,
        )
        self.last_result = result
        return result

    def reset(self):
        self._running_units = 0.0
        self.last_result = None


def test_counter_adapter_process_and_reset():
    adapter = CounterAdapter(config_path="unused.yaml", bridge=FakeBridge())
    req = SimpleNamespace(frame_index=3, original_depth=None)

    snap = adapter.process_keyframe(req)
    assert snap.running_units == 1.0
    assert snap.frame_index == 3
    assert adapter.processed_keyframes == 1

    frame_data, prepared, result = adapter.get_last_debug_bundle()
    assert frame_data == "frame"
    assert prepared == ["prepared"]
    assert result is not None

    adapter.reset()
    snap2 = adapter.get_snapshot()
    assert snap2.running_units == 0.0
    assert adapter.processed_keyframes == 0
