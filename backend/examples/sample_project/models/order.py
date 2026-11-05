from dataclasses import dataclass, field
from typing import List


@dataclass
class OrderItem:
    item_id: str
    name: str
    unit_price: float
    quantity: int

    def line_total(self) -> float:
        return self.unit_price * self.quantity


@dataclass
class Order:
    order_id: str
    user_id: str
    items: List[OrderItem] = field(default_factory=list)
    discount_code: str | None = None
    status: str = "pending"

    def subtotal(self) -> float:
        return sum(item.line_total() for item in self.items)
