from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import threading


@dataclass
class EventDecision:
    close_previous_order: bool = False
    previous_order_id: Optional[str] = None
    open_order: bool = False
    start_task: bool = False
    end_task: bool = False
    close_order: bool = False
    ignored: bool = False
    reason: str = ""


class InventoryEventManager:
    """
    Lightweight order/task state machine driven only by:
      - ITEM_START
      - ITEM_END
      - ORDER_END
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._order_id: Optional[str] = None
        self._task_active: bool = False
        self._task_ctx: Dict[str, object] = {}

    @property
    def order_id(self) -> Optional[str]:
        with self._lock:
            return self._order_id

    @property
    def task_active(self) -> bool:
        with self._lock:
            return self._task_active

    @property
    def task_context(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._task_ctx)

    def process_event(self, payload: Dict[str, object]) -> EventDecision:
        event_type = str(payload.get("event_type", "")).strip().upper()
        order_id = str(payload.get("order_id", "")).strip()

        with self._lock:
            if not event_type:
                return EventDecision(ignored=True, reason="missing_event_type")
            if not order_id:
                return EventDecision(ignored=True, reason="missing_order_id")

            if event_type == "ITEM_START":
                return self._on_item_start(order_id, payload)
            if event_type == "ITEM_END":
                return self._on_item_end(order_id)
            if event_type == "ORDER_END":
                return self._on_order_end(order_id)

            return EventDecision(ignored=True, reason="unsupported_event_type")

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            return {
                "order_id": self._order_id,
                "task_active": self._task_active,
                "task_context": dict(self._task_ctx),
            }

    def _on_item_start(self, order_id: str, payload: Dict[str, object]) -> EventDecision:
        hu_id = str(payload.get("hu_id", "")).strip()
        sku = str(payload.get("sku", "")).strip()
        quantity = int(payload.get("quantity", 0) or 0)

        if not hu_id or not sku:
            return EventDecision(ignored=True, reason="item_start_missing_hu_or_sku")

        decision = EventDecision()
        if self._order_id is None:
            self._order_id = order_id
            decision.open_order = True
        elif self._order_id != order_id:
            decision.close_previous_order = True
            decision.previous_order_id = self._order_id
            self._order_id = order_id
            decision.open_order = True

        if self._task_active:
            decision.end_task = True

        self._task_active = True
        self._task_ctx = {
            "order_id": order_id,
            "hu_id": hu_id,
            "sku": sku,
            "quantity": quantity,
            "description": str(payload.get("description", "")),
            "item_index": int(payload.get("item_index", 0) or 0),
            "hu_index": int(payload.get("hu_index", 0) or 0),
            "total_hus": int(payload.get("total_hus", 0) or 0),
        }
        decision.start_task = True
        return decision

    def _on_item_end(self, order_id: str) -> EventDecision:
        if self._order_id != order_id:
            return EventDecision(ignored=True, reason="item_end_order_mismatch")
        if not self._task_active:
            return EventDecision(ignored=True, reason="item_end_without_active_task")

        self._task_active = False
        self._task_ctx = {}
        return EventDecision(end_task=True)

    def _on_order_end(self, order_id: str) -> EventDecision:
        if self._order_id != order_id:
            return EventDecision(ignored=True, reason="order_end_order_mismatch")

        decision = EventDecision(close_order=True)
        if self._task_active:
            decision.end_task = True
            self._task_active = False
        self._task_ctx = {}
        self._order_id = None
        return decision
