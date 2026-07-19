"""
agents/risk_fraud_agent.py

QuoteFlow AI  — Risk & Fraud Assessor Agent

Reasoning Engine ONLY: Qwen2.5 (qwen-max via DashScope) analyzes the
transaction profile — combining the current RFQ's quote_amount with the
client's historical trust signals pulled deterministically from the
`clients` table — and returns a structured risk_score (0.0-1.0) plus a
reasoning_summary. The orchestrator's conditional edge (route_after_risk_assessment)
is what actually decides routing against risk_threshold; this module never
routes anything itself.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

import yaml
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# ============================================================
# Configuration
# ============================================================

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "settings.yaml",
)

with open(_CONFIG_PATH, "r", encoding="utf-8") as _handle:
    _SETTINGS = yaml.safe_load(_handle)

_QWEN_CFG = _SETTINGS["qwen"]
_MODEL_NAME = _QWEN_CFG["models"].get("risk_fraud_assessor", _QWEN_CFG["models"]["default"])
_MIN_TRUST_SCORE = _SETTINGS["thresholds"]["min_trust_score"]

_client = OpenAI(
    api_key=os.environ.get(_QWEN_CFG["api_key_env"], ""),
    base_url=_QWEN_CFG["base_url"],
    timeout=_QWEN_CFG["request_timeout_seconds"],
)

_SYSTEM_PROMPT = """You are the Risk & Fraud Assessor reasoning engine for QuoteFlow AI, a \
B2B quote-automation platform. You receive a transaction profile — the proposed quote \
amount, the client's historical trust signals, and RFQ metadata. Your ONLY job is to \
evaluate fraud/risk indicators and return a structured JSON risk assessment. You do NOT \
decide what happens next in the workflow (that is handled by deterministic routing logic \
downstream of you) — you only assess and explain.

Consider these risk signals:
- Low historical trust_score (below 0.3 is concerning, below 0.15 is severe).
- High flagged_count relative to total_orders (a high flag ratio is a strong risk signal).
- New or unknown clients (total_orders == 0) combined with an unusually large quote_amount \
  relative to typical B2B RFQ sizes.
- Missing or clearly disposable/suspicious-looking client_email domains.
- Any internal inconsistency between the RFQ text and the computed quote_amount.

Return a JSON object with EXACTLY these keys:
{
  "risk_score": number,          // 0.0 (no risk) to 1.0 (severe risk)
  "reasoning_summary": string    // 2-4 sentences explaining the specific signals driving the score
}

Rules:
- risk_score must be a well-calibrated float between 0.0 and 1.0, not a rounded bucket value.
- Be conservative: when historical data is sparse (new client, total_orders == 0), do not \
  default to a low risk_score purely due to lack of negative signal — treat data sparsity itself \
  as a mild risk factor.
- Return ONLY the JSON object. No markdown fences, no commentary, no preamble."""


class RiskAssessmentError(Exception):
    """Raised when the risk assessment cannot be completed or produces invalid data."""


def _get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_client_profile(conn: sqlite3.Connection, client_email: str | None) -> dict[str, Any]:
    """
    Retrieve historical trust signals for the client. If the client has no
    prior record, returns a neutral-but-flagged-as-new profile rather than
    raising, since first-time clients are a normal and expected case.
    """
    if not client_email:
        return {
            "known_client": False,
            "trust_score": _MIN_TRUST_SCORE,
            "total_orders": 0,
            "flagged_count": 0,
        }

    cursor = conn.execute(
        """
        SELECT trust_score, total_orders, flagged_count
        FROM clients
        WHERE client_email = ?
        """,
        (client_email,),
    )
    row = cursor.fetchone()
    if row is None:
        return {
            "known_client": False,
            "trust_score": 0.5,  # neutral default per clients table schema default
            "total_orders": 0,
            "flagged_count": 0,
        }

    return {
        "known_client": True,
        "trust_score": row["trust_score"],
        "total_orders": row["total_orders"],
        "flagged_count": row["flagged_count"],
    }


def _call_qwen_with_retry(transaction_profile: dict[str, Any]) -> dict[str, Any]:
    """Call Qwen (qwen-max) with retry/backoff, enforcing JSON-object output mode."""
    max_retries = _QWEN_CFG["max_retries"]
    backoff = _QWEN_CFG["retry_backoff_seconds"]
    last_error: Exception | None = None

    user_message = (
        "TRANSACTION PROFILE:\n\n"
        f"{json.dumps(transaction_profile, indent=2, default=str)}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = _client.chat.completions.create(
                model=_MODEL_NAME,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=_QWEN_CFG["generation"]["temperature"],
                top_p=_QWEN_CFG["generation"]["top_p"],
                max_tokens=_QWEN_CFG["generation"]["max_tokens"],
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            return json.loads(raw_content)
        except (APITimeoutError, RateLimitError, APIError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff * attempt)
                continue

    raise RiskAssessmentError(
        f"[risk_fraud_agent] Qwen call failed after {max_retries} attempts: {last_error}"
    )


def assess_risk(state: dict[str, Any], db_path: str) -> dict[str, Any]:
    """
    Assess fraud/transaction risk for the current RFQ using Qwen (qwen-max),
    grounded in deterministic historical client data pulled from SQLite.

    Args:
        state: The current AgentState dict. Must contain 'quote_amount'.
               May contain 'client_email', 'client_name', 'extracted_items'.
        db_path: Absolute path to the QuoteFlow AI SQLite database.

    Returns:
        A dict with keys: risk_score (float, 0.0-1.0), reasoning_summary (str).

    Raises:
        RiskAssessmentError: if assessment fails after retries or produces invalid data.
    """
    quote_amount = state.get("quote_amount")
    if quote_amount is None:
        raise RiskAssessmentError("[risk_fraud_agent] 'quote_amount' is missing — pricing must run before risk assessment")

    conn = _get_connection(db_path)
    try:
        client_profile = _fetch_client_profile(conn, state.get("client_email"))
    except sqlite3.Error as exc:
        raise RiskAssessmentError(f"[risk_fraud_agent] Database error fetching client profile: {exc}") from exc
    finally:
        conn.close()

    transaction_profile = {
        "client_name": state.get("client_name"),
        "client_email": state.get("client_email"),
        "quote_amount": quote_amount,
        "margin_pct": state.get("margin_pct"),
        "line_item_count": len(state.get("extracted_items", [])),
        "client_history": client_profile,
    }

    parsed = _call_qwen_with_retry(transaction_profile)

    risk_score = parsed.get("risk_score")
    try:
        risk_score = float(risk_score)
        risk_score = max(0.0, min(1.0, risk_score))
    except (TypeError, ValueError):
        raise RiskAssessmentError(f"[risk_fraud_agent] Qwen returned a non-numeric risk_score: {risk_score!r}")

    reasoning_summary = str(parsed.get("reasoning_summary", "")).strip()
    if not reasoning_summary:
        raise RiskAssessmentError("[risk_fraud_agent] Qwen returned an empty reasoning_summary")

    return {
        "risk_score": risk_score,
        "reasoning_summary": reasoning_summary,
    }