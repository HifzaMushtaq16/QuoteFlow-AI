"""
agents/pricing_agent.py

QuoteFlow AI — Pricing Calculator Tool

Deterministic tool node — NO Qwen / LLM call happens here. This module reads
base_unit_price and margin bounds from the `pricing_rules` SQLite table and
computes the total quote_amount and the blended margin_pct for the RFQ's
extracted line items. Pure, reproducible arithmetic only.
"""

from __future__ import annotations

import sqlite3
from typing import Any


class PricingError(Exception):
    """Raised when pricing calculation cannot be completed against known rules."""


# Fallback rule applied when an item_sku has no matching row in pricing_rules.
# Kept intentionally conservative (higher margin, no discount) so unknown
# SKUs never under-price a quote.
_FALLBACK_RULE = {
    "base_unit_price": 100.00,
    "min_margin_pct": 0.20,
    "max_discount_pct": 0.0,
}


def _get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_pricing_rule(conn: sqlite3.Connection, item_sku: str) -> dict[str, Any]:
    """
    Look up the most recently updated pricing_rules row for a given SKU.
    Falls back to a conservative default rule if no match is found.
    """
    cursor = conn.execute(
        """
        SELECT base_unit_price, min_margin_pct, max_discount_pct
        FROM pricing_rules
        WHERE item_sku = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (item_sku,),
    )
    row = cursor.fetchone()
    if row is None:
        return dict(_FALLBACK_RULE)
    return {
        "base_unit_price": row["base_unit_price"],
        "min_margin_pct": row["min_margin_pct"],
        "max_discount_pct": row["max_discount_pct"],
    }


def calculate_quote(items: list[dict[str, Any]], db_path: str) -> tuple[float, float]:
    """
    Compute the total quote_amount and blended margin_pct for a set of RFQ line items.

    For each item:
        line_base_cost   = base_unit_price * quantity
        line_priced_total = line_base_cost * (1 + min_margin_pct)

    The final margin_pct returned is the quantity-weighted average of each
    line's min_margin_pct, giving an accurate blended margin across a
    multi-item RFQ rather than a naive arithmetic mean.

    Args:
        items: List of dicts like {"item_sku": str, "quantity": float, ...}
               as produced by agents.intake_parser_agent.run_intake_parser.
        db_path: Absolute path to the QuoteFlow AI SQLite database.

    Returns:
        (quote_amount, margin_pct) — quote_amount rounded to 2 decimals,
        margin_pct rounded to 4 decimals (e.g. 0.1234 == 12.34%).

    Raises:
        PricingError: if 'items' is empty or malformed.
    """
    if not items:
        raise PricingError("[pricing_agent] No line items provided — cannot calculate quote")

    conn = _get_connection(db_path)
    try:
        total_base_cost = 0.0
        total_priced_amount = 0.0
        weighted_margin_numerator = 0.0

        for item in items:
            item_sku = str(item.get("item_sku", "")).strip().upper()
            if not item_sku:
                raise PricingError(f"[pricing_agent] Line item missing 'item_sku': {item}")

            try:
                quantity = float(item.get("quantity", 1))
            except (TypeError, ValueError):
                raise PricingError(f"[pricing_agent] Invalid quantity for item {item_sku}: {item.get('quantity')}")

            if quantity <= 0:
                raise PricingError(f"[pricing_agent] Quantity must be positive for item {item_sku}")

            rule = _fetch_pricing_rule(conn, item_sku)
            line_base_cost = rule["base_unit_price"] * quantity
            line_priced_total = line_base_cost * (1.0 + rule["min_margin_pct"])

            total_base_cost += line_base_cost
            total_priced_amount += line_priced_total
            weighted_margin_numerator += rule["min_margin_pct"] * line_base_cost

        if total_base_cost <= 0:
            raise PricingError("[pricing_agent] Computed base cost is zero — check pricing_rules data")

        blended_margin_pct = weighted_margin_numerator / total_base_cost
        quote_amount = round(total_priced_amount, 2)
        margin_pct = round(blended_margin_pct, 4)

        return quote_amount, margin_pct
    except sqlite3.Error as exc:
        raise PricingError(f"[pricing_agent] Database error during pricing calculation: {exc}") from exc
    finally:
        conn.close()