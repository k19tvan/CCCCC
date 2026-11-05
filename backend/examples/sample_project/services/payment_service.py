"""
Payment service: handles transactions and external payment gateway calls.
"""

from services.order_service import OrderService
from models.order import Order


class PaymentService:
    def __init__(self, order_service: OrderService):
        self.order_service = order_service

    def process_payment(self, order: Order, payment_method: str) -> bool:
        amount = self.order_service.calculate_total(order)
        if amount <= 0:
            raise ValueError(f"Invalid payment amount: {amount}. Amount must be positive.")

        print(f"[PaymentGateway] Charging {amount} via {payment_method} for Order {order.order_id}")
        order.status = "paid"
        return True
