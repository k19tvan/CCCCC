"""
calculator.py - simple calculation module for the sample repo fixture.
Used to test the list_files, read_file and search_code tools.
"""


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers. Raises ZeroDivisionError if b == 0."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def calculate_total(items: list[float]) -> float:
    """
    Sum a list of numbers.
    Returns 0.0 for an empty list.
    """
    if not items:
        return 0.0
    return sum(items)


def calculate_average(items: list[float]) -> float:
    """Average a list of numbers. Raises ValueError if empty."""
    if not items:
        raise ValueError("Cannot compute the average of an empty list")
    return sum(items) / len(items)


def get_discount_rate(price: float, discount_percent: float) -> float:
    """Return the price after a discount (discount_percent: 0-100)."""
    if not 0 <= discount_percent <= 100:
        raise ValueError(f"discount_percent must be between 0 and 100, got: {discount_percent}")
    return price * (1 - discount_percent / 100)
