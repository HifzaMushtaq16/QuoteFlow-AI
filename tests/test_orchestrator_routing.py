"""
tests/test_orchestrator_routing.py

Unit tests for the pure conditional-edge routing functions in
backend/orchestrator.py. These functions decide LangGraph's next node
based on AgentState — they contain no I/O, no Qwen calls, and no database
access, making them straightforward to test in isolation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.orchestrator import (
    route_after_human_approval,
    route_after_negotiation,
    route_after_risk_assessment,
)


def test_low_risk_routes_to_negotiation_agent():
    state = {"status": "running", "risk_score": 0.10}
    assert route_after_risk_assessment(state) == "negotiation_agent"


def test_high_risk_routes_to_human_approval_gate():
    state = {"status": "running", "risk_score": 0.85}
    assert route_after_risk_assessment(state) == "human_approval_gate"


def test_risk_exactly_at_threshold_routes_to_human_approval_gate():
    """risk_score >= risk_threshold (0.6) must route to human review, not negotiation."""
    state = {"status": "running", "risk_score": 0.6}
    assert route_after_risk_assessment(state) == "human_approval_gate"


def test_risk_assessment_error_status_routes_to_error_end():
    state = {"status": "error", "risk_score": None}
    assert route_after_risk_assessment(state) == "error_end"


def test_high_confidence_routes_to_pdf_quote_generator():
    state = {"status": "running", "confidence_score": 0.90}
    assert route_after_negotiation(state) == "pdf_quote_generator"


def test_low_confidence_routes_to_human_approval_gate():
    state = {"status": "running", "confidence_score": 0.40}
    assert route_after_negotiation(state) == "human_approval_gate"


def test_negotiation_error_status_routes_to_error_end():
    state = {"status": "error", "confidence_score": None}
    assert route_after_negotiation(state) == "error_end"


def test_approved_decision_routes_to_pdf_quote_generator():
    state = {"status": "approved"}
    assert route_after_human_approval(state) == "pdf_quote_generator"


def test_rejected_decision_routes_to_end():
    state = {"status": "rejected"}
    assert route_after_human_approval(state) == "end"


def test_human_approval_error_status_routes_to_error_end():
    state = {"status": "error"}
    assert route_after_human_approval(state) == "error_end"