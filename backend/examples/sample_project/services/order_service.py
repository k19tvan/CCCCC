"""
Order service: handles discount calculation, order pricing, and checkout logic.
"""

from typing import Optional
from models.order import Order
import config


class OrderService:
    def __init__(self):
        self.tax_rate = config.DEFAULT_TAX_RATE

    def apply_discount(self, subtotal: float, discount_code: Optional[str]) -> float:
        """
        Calculate the amount after applying a discount code.
        BUG: The subtotal after discount is not checked, so the total goes negative
        when a fixed-amount (FIXED) discount is larger than the order total!
        """
        if not discount_code:
            return subtotal

        code = discount_code.upper().strip()
        if code == "SUMMER20":
            return subtotal * 0.8
        elif code == "VIP50":
            return subtotal * 0.5
        elif code == "FIXED100K":
            # BUG: If subtotal < 100000, the result is negative!
            discounted = subtotal - 100000.0
            return discounted

        raise ValueError(f"Invalid discount code: {discount_code}")

    def calculate_total(self, order: Order) -> float:
        """
        Calculate the final payment total of the order, including tax and discount.
        """
        sub = order.subtotal()
        after_discount = self.apply_discount(sub, order.discount_code)
        tax = after_discount * self.tax_rate
        total = after_discount + tax
        return round(total, 2)
