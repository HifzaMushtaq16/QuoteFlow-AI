"""
agents/intake_parser_agent.py

QuoteFlow AI  — Intake Parser Agent

Reasoning Engine ONLY: Qwen2.5 (qwen-plus via DashScope's OpenAI-compatible
endpoint) is used strictly to extract structured RFQ data from raw inbound
text. It never decides graph routing — it returns a JSON object which the
orchestrator node shell (backend/orchestrator.py::intake_parser_node) maps
directly onto AgentState fields.
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
_MODEL_NAME = _QWEN_CFG["models"].get("intake_parser", _QWEN_CFG["models"]["default"])

_client = OpenAI(
    api_key=os.environ.get(_QWEN_CFG["api_key_env"], ""),
    base_url=_QWEN_CFG["base_url"],
    timeout=_QWEN_CFG["request_timeout_seconds"],
)

_SYSTEM_PROMPT = """You are the Intake Parser reasoning engine for QuoteFlow AI, a B2B \
quote-automation platform. You will receive raw inbound RFQ (Request for Quote) text \
(from an email or document). Your ONLY job is to extract structured data and return it \
as a single, strictly valid JSON object. You do NOT make business decisions, you do NOT \
calculate prices, and you do NOT decide what happens next in the workflow — you only \
extract and structure information that is present or clearly implied in the text.

Return a JSON object with EXACTLY these keys:
{
  "client_name": string or null,
  "client_email": string or null,
  "extracted_items": [
    {
      "item_sku": string,          // best-guess canonical SKU token, uppercase, hyphenated
      "description": string,
      "quantity": number
    }
  ],
  "confidence_score": number,      // 0.0-1.0, your confidence in the extraction accuracy
  "reasoning_summary": string      // 1-3 sentences explaining your extraction choices
}

Rules:
- If client_email is not explicitly present, return null (do not invent one).
- quantity must always be a positive number; if not stated, infer 1.
- item_sku should be a short uppercase token (e.g. "SKU-CLOUD-COMPUTE-STD") built from the \
  clearest product/service mentioned. If several plausible SKUs exist, pick the closest match \
  to a standard cloud/enterprise services catalog term.
- confidence_score should reflect ambiguity: lower it if the text is vague, contradictory, or \
  missing key fields.
- Return ONLY the JSON object. No markdown fences, no commentary, no preamble."""


class IntakeParserError(Exception):
    """Raised when the intake parser cannot produce a valid structured result."""


def _call_qwen_with_retry(rfq_raw_text: str) -> dict[str, Any]:
    """
    Call Qwen (qwen-plus) with retry/backoff, enforcing JSON-object output mode.

    Raises:
        IntakeParserError: if all retries are exhausted or the response is not valid JSON.
    """
    max_retries = _QWEN_CFG["max_retries"]
    backoff = _QWEN_CFG["retry_backoff_seconds"]
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = _client.chat.completions.create(
                model=_MODEL_NAME,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"RFQ TEXT:\n\n{rfq_raw_text}"},
                ],
                temperature=_QWEN_CFG["generation"]["temperature"],
                top_p=_QWEN_CFG["generation"]["top_p"],
                max_tokens=_QWEN_CFG["generation"]["max_tokens"],
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)
            return parsed
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

    raise IntakeParserError(
        f"[intake_parser_agent] Qwen call failed after {max_retries} attempts: {last_error}"
    )


def _validate_and_normalize(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure the parsed JSON matches the expected shape; fill safe defaults where possible."""
    normalized: dict[str, Any] = {
        "client_name": parsed.get("client_name"),
        "client_email": parsed.get("client_email"),
        "extracted_items": [],
        "confidence_score": None,
        "reasoning_summary": parsed.get("reasoning_summary", ""),
    }

    raw_items = parsed.get("extracted_items", [])
    if not isinstance(raw_items, list):
        raise IntakeParserError("[intake_parser_agent] 'extracted_items' is not a list")

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        sku = str(raw_item.get("item_sku", "")).strip().upper()
        description = str(raw_item.get("description", "")).strip()
        try:
            quantity = float(raw_item.get("quantity", 1))
        except (TypeError, ValueError):
            quantity = 1.0
        if quantity <= 0:
            quantity = 1.0
        if sku:
            normalized["extracted_items"].append({
                "item_sku": sku,
                "description": description or sku,
                "quantity": quantity,
            })

    if not normalized["extracted_items"]:
        raise IntakeParserError("[intake_parser_agent] No valid line items could be extracted from RFQ text")

    confidence = parsed.get("confidence_score")
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5  # conservative fallback, never silently defaults to high confidence

    normalized["confidence_score"] = confidence
    return normalized


def run_intake_parser(state: dict[str, Any]) -> dict[str, Any]:
    """
    Extract structured RFQ data (client info + line items) from raw text using Qwen.

    Args:
        state: The current AgentState dict. Must contain 'rfq_raw_text'.

    Returns:
        A dict with keys: client_name, client_email, extracted_items,
        confidence_score, reasoning_summary — ready to be merged into AgentState
        by the orchestrator's intake_parser_node.

    Raises:
        IntakeParserError: if extraction fails after retries or produces invalid data.
    """
    rfq_raw_text = state.get("rfq_raw_text", "").strip()
    if not rfq_raw_text:
        raise IntakeParserError("[intake_parser_agent] 'rfq_raw_text' is empty — nothing to parse")

    parsed = _call_qwen_with_retry(rfq_raw_text)
    normalized = _validate_and_normalize(parsed)
    return normalized