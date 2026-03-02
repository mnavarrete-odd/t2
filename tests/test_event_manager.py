from lib.inventory.event_manager import InventoryEventManager


def test_item_start_opens_order_and_starts_task():
    mgr = InventoryEventManager()
    decision = mgr.process_event(
        {
            "event_type": "ITEM_START",
            "order_id": "ORD-1",
            "hu_id": "HU-1",
            "sku": "SKU-1",
            "quantity": 2,
        }
    )
    assert not decision.ignored
    assert decision.open_order
    assert decision.start_task
    assert mgr.order_id == "ORD-1"
    assert mgr.task_active


def test_item_end_without_task_is_ignored():
    mgr = InventoryEventManager()
    mgr.process_event(
        {
            "event_type": "ITEM_START",
            "order_id": "ORD-1",
            "hu_id": "HU-1",
            "sku": "SKU-1",
            "quantity": 2,
        }
    )
    mgr.process_event({"event_type": "ITEM_END", "order_id": "ORD-1", "hu_id": "HU-1"})
    decision = mgr.process_event(
        {"event_type": "ITEM_END", "order_id": "ORD-1", "hu_id": "HU-1"}
    )
    assert decision.ignored
    assert decision.reason == "item_end_without_active_task"


def test_order_switch_requests_previous_close():
    mgr = InventoryEventManager()
    mgr.process_event(
        {
            "event_type": "ITEM_START",
            "order_id": "ORD-1",
            "hu_id": "HU-1",
            "sku": "SKU-1",
            "quantity": 1,
        }
    )
    decision = mgr.process_event(
        {
            "event_type": "ITEM_START",
            "order_id": "ORD-2",
            "hu_id": "HU-2",
            "sku": "SKU-2",
            "quantity": 1,
        }
    )
    assert decision.close_previous_order
    assert decision.previous_order_id == "ORD-1"
    assert decision.start_task
    assert mgr.order_id == "ORD-2"


def test_order_end_closes_task_and_order():
    mgr = InventoryEventManager()
    mgr.process_event(
        {
            "event_type": "ITEM_START",
            "order_id": "ORD-1",
            "hu_id": "HU-1",
            "sku": "SKU-1",
            "quantity": 1,
        }
    )
    decision = mgr.process_event({"event_type": "ORDER_END", "order_id": "ORD-1"})
    assert decision.close_order
    assert decision.end_task
    assert mgr.order_id is None
    assert not mgr.task_active
