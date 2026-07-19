"""
tests/test_pricing_agent.py

Unit tests for agents/pricing_agent.py — the deterministic pricing tool
(no Qwen/LLM calls). Uses a throwaway in-memory-style SQLite file seeded
with known pricing_rules so results are fully predictable.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.pricing_agent import PricingError, calculate_quote


@pytest.fixture
def seeded_db_path():
    """Create a temporary SQLite file with a minimal pricing_rules table."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE pricing_rules (
            rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_sku TEXT NOT NULL,
            base_unit_price REAL NOT NULL,
            min_margin_pct REAL NOT NULL DEFAULT 0.10,
            max_discount_pct REAL NOT NULL DEFAULT 0.15,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "INSERT INTO pricing_rules (item_sku, base_unit_price, min_margin_pct, max_discount_pct) "
        "VALUES ('SKU-CLOUD-COMPUTE-STD', 100.00, 0.20, 0.10)"
    )
    conn.execute(
        "INSERT INTO pricing_rules (item_sku, base_unit_price, min_margin_pct, max_discount_pct) "
        "VALUES ('SKU-SUPPORT-ENTERPRISE', 500.00, 0.10, 0.05)"
    )
    conn.commit()
    conn.close()

    yield path
    os.remove(path)


def test_single_item_quote_matches_expected_margin(seeded_db_path):
    """base_unit_price * qty * (1 + margin) should compute exactly for one line item."""
    items = [{"item_sku": "SKU-CLOUD-COMPUTE-STD", "quantity": 10}]
    quote_amount, margin_pct = calculate_quote(items, seeded_db_path)

    # 100 * 10 = 1000 base cost; priced = 1000 * 1.20 = 1200
    assert quote_amount == 1200.00
    assert margin_pct == 0.20


def test_multi_item_quote_uses_weighted_average_margin(seeded_db_path):
    """Blended margin should be weighted by each line's base cost, not a flat average."""
    items = [
        {"item_sku": "SKU-CLOUD-COMPUTE-STD", "quantity": 10},   # base cost 1000, margin 0.20
        {"item_sku": "SKU-SUPPORT-ENTERPRISE", "quantity": 1},   # base cost 500, margin 0.10
    ]
    quote_amount, margin_pct = calculate_quote(items, seeded_db_path)

    # total base cost = 1500; weighted margin = (1000*0.20 + 500*0.10) / 1500 = 0.1667
    assert quote_amount == pytest.approx(1200.00 + 550.00, abs=0.01)
    assert margin_pct == pytest.approx(0.1667, abs=0.001)


def test_unknown_sku_falls_back_to_conservative_default(seeded_db_path):
    """An unrecognized SKU should use the conservative fallback rule, not crash."""
    items = [{"item_sku": "SKU-DOES-NOT-EXIST", "quantity": 5}]
    quote_amount, margin_pct = calculate_quote(items, seeded_db_path)

    assert quote_amount > 0
    assert margin_pct == 0.20  # matches _FALLBACK_RULE in pricing_agent.py


def test_empty_items_list_raises_pricing_error(seeded_db_path):
    with pytest.raises(PricingError):
        calculate_quote([], seeded_db_path)


def test_zero_or_negative_quantity_raises_pricing_error(seeded_db_path):
    items = [{"item_sku": "SKU-CLOUD-COMPUTE-STD", "quantity": 0}]
    with pytest.raises(PricingError):
        calculate_quote(items, seeded_db_path)


def test_missing_sku_field_raises_pricing_error(seeded_db_path):
    items = [{"quantity": 5}]
    with pytest.raises(PricingError):
        calculate_quote(items, seeded_db_path)