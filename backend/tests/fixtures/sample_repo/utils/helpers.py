"""helpers.py — Utility functions."""

def format_price(price: float, currency: str = "VND") -> str:
    """Format giá ti?n."""
    return f"{price:,.0f} {currency}"

def slugify(text: str) -> str:
    """Convert text thành slug."""
    return text.lower().replace(" ", "-")
