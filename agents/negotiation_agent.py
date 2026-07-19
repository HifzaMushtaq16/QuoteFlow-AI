"""
agents/negotiation_agent.py

QuoteFlow AI v3.0 — Negotiation Agent

Reasoning Engine ONLY: Qwen2.5 (qwen-max via DashScope) evaluates the current
quote against margin floors and discount ceilings pulled from pricing_rules,
and proposes an optimized quote_amount plus a confidence_score reflecting how
confident it is that the counter-offer will be accepted without further human
review. LangGraph's route_after_negotiation conditional edge is solely
responsible for deciding whether to proceed to pdf_quote_generator or escalate
to human_approval_gate based on that confidence_score against
confidence_threshold — this module never makes that routing decision itself.
"""

from __future__ import annotations

import json
import os
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
_MODEL_NAME = _QWEN_CFG["models"].get("negotiation_agent", _QWEN_CFG["models"]["default"])
_MAX_NEGOTIATION_ROUNDS = _SETTINGS["thresholds"]["max_negotiation_rounds"]

_client = OpenAI(
    api_key=os.environ.get(_QWEN_CFG["api_key_env"], ""),
    base_url=_QWEN_CFG["base_url"],
    timeout=_QWEN_CFG["request_timeout_seconds"],
)

_SYSTEM_PROMPT = """You are the Negotiation reasoning engine for QuoteFlow AI, a B2B \
quote-automation platform. You receive the current quote_amount, its blended margin_pct, \
and how many negotiation rounds have already occurred for this RFQ. Your job is to decide \
whether the current quote should be adjusted (e.g. a modest discount to close the deal \
faster) while respecting margin floors, and to return your confidence that this quote is \
ready to be finalized without further human review. You do NOT decide workflow routing \
yourself — you only propose numbers and a confidence assessment.

Return a JSON object with EXACTLY these keys:
{
  "quote_amount": number,        // the (possibly adjusted) final quote amount
  "confidence_score": number,    // 0.0-1.0, your confidence this quote is ready to finalize
  "reasoning_summary": string    // 2-4 sentences explaining any adjustment and the confidence rationale
}

Rules:
- Never propose a quote_amount that would reduce margin_pct below the floor already reflected \
  in the input (you are given the ALREADY-margin-inclusive quote_amount; do not discount below \
  the point where margin would go negative).
- If negotiation_rounds is already at or beyond max_negotiation_rounds, do not propose further \
  discounts — hold the quote_amount steady and set confidence_score based on whether the current \
  number is reasonable to finalize as-is.
- Higher negotiation_rounds with no convergence should lower your confidence_score, since repeated \
  rounds signal friction that likely needs human judgment.
- A small, well-justified discount (never exceeding standard B2B norms of a few percentage points) \
  paired with a clear reasoning_summary can raise confidence_score, since it reflects reasonable \
  deal-closing behavior.
- Return ONLY the JSON object. No markdown fences, no commentary, no preamble."""


class NegotiationError(Exception):
    """Raised when negotiation cannot be completed or produces invalid data."""


def _call_qwen_with_retry(negotiation_context: dict[str, Any]) -> dict[str, Any]:
    """Call Qwen (qwen-max) with retry/backoff, enforcing JSON-object output mode."""
    max_retries = _QWEN_CFG["max_retries"]
    backoff = _QWEN_CFG["retry_backoff_seconds"]
    last_error: Exception | None = None

    user_message = (
        "NEGOTIATION CONTEXT:\n\n"
        f"{json.dumps(negotiation_context, indent=2, default=str)}"
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

    raise NegotiationError(
        f"[negotiation_agent] Qwen call failed after {max_retries} attempts: {last_error}"
    )


def run_negotiation(state: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate and (optionally) adjust the current quote via Qwen (qwen-max),
    returning an updated quote_amount and a confidence_score used by the
    orchestrator's route_after_negotiation conditional edge.

    Args:
        state: The current AgentState dict. Must contain 'quote_amount' and
               'margin_pct'. May contain 'negotiation_rounds'.

    Returns:
        A dict with keys: quote_amount (float), confidence_score (float, 0.0-1.0),
        reasoning_summary (str).

    Raises:
        NegotiationError: if negotiation fails after retries or produces invalid data.
    """
    quote_amount = state.get("quote_amount")
    margin_pct = state.get("margin_pct")

    if quote_amount is None or margin_pct is None:
        raise NegotiationError(
            "[negotiation_agent] 'quote_amount' and 'margin_pct' must be set before negotiation — "
            "pricing_calculator must run first"
        )

    negotiation_rounds = state.get("negotiation_rounds", 1)

    negotiation_context = {
        "quote_amount": quote_amount,
        "margin_pct": margin_pct,
        "negotiation_rounds": negotiation_rounds,
        "max_negotiation_rounds": _MAX_NEGOTIATION_ROUNDS,
        "risk_score": state.get("risk_score"),
        "line_item_count": len(state.get("extracted_items", [])),
    }

    parsed = _call_qwen_with_retry(negotiation_context)

    proposed_amount = parsed.get("quote_amount")
    try:
        proposed_amount = float(proposed_amount)
    except (TypeError, ValueError):
        raise NegotiationError(f"[negotiation_agent] Qwen returned a non-numeric quote_amount: {proposed_amount!r}")

    if proposed_amount <= 0:
        raise NegotiationError(f"[negotiation_agent] Qwen proposed an invalid quote_amount: {proposed_amount}")

    # Safety floor: never allow a proposed amount below the original base cost
    # implied by the pre-negotiation margin_pct — protects against a
    # miscalibrated LLM response eroding margin below zero.
    minimum_allowed_amount = quote_amount / (1.0 + margin_pct) if margin_pct > 0 else quote_amount
    if proposed_amount < minimum_allowed_amount:
        proposed_amount = minimum_allowed_amount

    confidence_score = parsed.get("confidence_score")
    try:
        confidence_score = float(confidence_score)
        confidence_score = max(0.0, min(1.0, confidence_score))
    except (TypeError, ValueError):
        raise NegotiationError(f"[negotiation_agent] Qwen returned a non-numeric confidence_score: {confidence_score!r}")

    reasoning_summary = str(parsed.get("reasoning_summary", "")).strip()
    if not reasoning_summary:
        raise NegotiationError("[negotiation_agent] Qwen returned an empty reasoning_summary")

    return {
        "quote_amount": round(proposed_amount, 2),
        "confidence_score": confidence_score,
        "reasoning_summary": reasoning_summary,
    }