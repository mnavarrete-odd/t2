from lib.inventory.event_manager import InventoryEventManager


def test_full_signal_cycle_and_repeat():
    mgr = InventoryEventManager()

    d1 = mgr.process_event(
        {
            "event_type": "ITEM_START",
            "order_id": "ORD-100",
            "hu_id": "HU-1",
            "sku": "SKU-1",
            "quantity": 1,
        }
    )
    assert d1.open_order
    assert d1.start_task

    d2 = mgr.process_event({"event_type": "ITEM_END", "order_id": "ORD-100", "hu_id": "HU-1"})
    assert d2.end_task
    assert not mgr.task_active

    d3 = mgr.process_event(
        {
            "event_type": "ITEM_START",
            "order_id": "ORD-100",
            "hu_id": "HU-1",
            "sku": "SKU-2",
            "quantity": 2,
        }
    )
    assert d3.start_task
    assert mgr.task_active

    d4 = mgr.process_event({"event_type": "ORDER_END", "order_id": "ORD-100"})
    assert d4.close_order
    assert d4.end_task
    assert mgr.order_id is None
    assert not mgr.task_active
