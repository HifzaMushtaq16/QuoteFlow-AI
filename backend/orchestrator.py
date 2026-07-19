"""
backend/orchestrator.py

QuoteFlow AI  — LangGraph Orchestrator (State Manager & Control Flow)

Strict separation of concerns:
    - Qwen2.5 (called from within each node) is ONLY a reasoning engine.
      It returns structured JSON: {decision, confidence_score, reasoning_summary}.
    - LangGraph's StateGraph is the ONLY component that decides which node
      executes next. Conditional edges read the structured JSON / deterministic
      calculations and route accordingly.
    - Every node transition is mirrored into the SQLite `audit_log` and
      `task_state` tables, which are the single source of truth queried
      live by dashboard.py.

This module defines:
    1. AgentState  — the typed schema flowing through the graph.
    2. Node shells  — intake_parser, pricing_calculator, risk_fraud_assessor,
       negotiation_agent, human_approval_gate, pdf_quote_generator.
    3. Conditional edge routers.
    4. The compiled, checkpointed StateGraph (SqliteSaver-backed).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional, TypedDict

import yaml
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
load_dotenv()

# ============================================================
# Configuration loading
# ============================================================

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "settings.yaml",
)


def load_settings(config_path: str = _CONFIG_PATH) -> dict:
    """
    Load and return the QuoteFlow AI settings.yaml as a dict.

    Raises:
        FileNotFoundError: if the settings file does not exist.
        yaml.YAMLError: if the file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[orchestrator] settings.yaml not found at expected path: {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


SETTINGS: dict = load_settings()

CONFIDENCE_THRESHOLD: float = SETTINGS["thresholds"]["confidence_threshold"]
RISK_THRESHOLD: float = SETTINGS["thresholds"]["risk_threshold"]
DB_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    SETTINGS["database"]["sqlite_path"],
)


# ============================================================
# AgentState schema
# ============================================================

class AgentState(TypedDict, total=False):
    """
    The single typed state object threaded through every LangGraph node.

    Mirrors task_state / audit_log columns so that persisting state to
    SQLite after each node execution is a direct, lossless mapping.
    """

    # Identity / threading
    task_id: str
    thread_id: str
    step_index: int

    # RFQ intake
    client_name: Optional[str]
    client_email: Optional[str]
    rfq_raw_text: str
    extracted_items: list[dict[str, Any]]

    # Pricing
    quote_amount: Optional[float]
    margin_pct: Optional[float]

    # Risk & negotiation
    risk_score: Optional[float]
    confidence_score: Optional[float]
    reasoning_summary: Optional[str]
    negotiation_rounds: int

    # Human-in-the-loop
    human_decision: Optional[Literal["approved", "rejected"]]
    human_reviewer: Optional[str]

    # Output
    pdf_oss_url: Optional[str]

    # Control flow bookkeeping
    current_node: str
    next_node: Optional[str]
    status: Literal[
        "running", "pending_approval", "approved", "rejected", "finalized", "error"
    ]
    error_message: Optional[str]


# ============================================================
# Persistence helpers — single source of truth writers
# ============================================================

def _get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with row-level access and FK enforcement."""
    conn = sqlite3.connect(DB_PATH, timeout=SETTINGS["database"]["connection_timeout_seconds"])
    conn.execute("PRAGMA foreign_keys = ON;")
    if SETTINGS["database"].get("enable_wal_mode"):
        conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_task_exists(state: AgentState) -> None:
    """
    Idempotently insert the task_state row for a new RFQ thread.
    Safe to call multiple times; uses INSERT OR IGNORE on task_id.
    """
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO task_state (
                task_id, thread_id, client_name, client_email,
                rfq_raw_text, current_node, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state["task_id"],
                state["thread_id"],
                state.get("client_name"),
                state.get("client_email"),
                state.get("rfq_raw_text"),
                state.get("current_node", "intake_parser"),
                state.get("status", "running"),
            ),
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise RuntimeError(f"[orchestrator] Failed to initialize task_state: {exc}") from exc
    finally:
        conn.close()


def update_task_state(state: AgentState) -> None:
    """
    Update the mutable columns of task_state after a node finishes executing.
    This is the live row dashboard.py polls for 'current active node'.
    """
    conn = _get_connection()
    try:
        conn.execute(
            """
            UPDATE task_state
            SET current_node       = ?,
                status              = ?,
                client_name         = COALESCE(?, client_name),
                client_email        = COALESCE(?, client_email),
                latest_confidence   = ?,
                latest_risk_score   = ?,
                quote_amount        = ?,
                margin_pct          = ?,
                pdf_oss_url         = ?,
                updated_at          = ?
            WHERE task_id = ?
            """,
            (
                state.get("current_node"),
                state.get("status", "running"),
                state.get("client_name"),
                state.get("client_email"),
                state.get("confidence_score"),
                state.get("risk_score"),
                state.get("quote_amount"),
                state.get("margin_pct"),
                state.get("pdf_oss_url"),
                datetime.now(timezone.utc).isoformat(),
                state["task_id"],
            ),
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise RuntimeError(f"[orchestrator] Failed to update task_state: {exc}") from exc
    finally:
        conn.close()
def write_audit_log(
    state: AgentState,
    node_name: str,
    agent_name: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    latency_ms: int,
    next_node: Optional[str] = None,
    tool_call_name: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Append-only write to audit_log. This is the authoritative ledger the
    Streamlit dashboard reconstructs live node visibility and trend charts
    from — every node execution, success or failure, MUST call this.
    """
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO audit_log (
                task_id, thread_id, node_name, agent_name, step_index,
                input_payload, output_payload, reasoning_summary,
                confidence_score, risk_score, next_node,
                tool_call_name, latency_ms, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state["task_id"],
                state["thread_id"],
                node_name,
                agent_name,
                state.get("step_index", 0),
                json.dumps(input_payload, default=str),
                json.dumps(output_payload, default=str),
                state.get("reasoning_summary"),
                state.get("confidence_score"),
                state.get("risk_score"),
                next_node,
                tool_call_name,
                latency_ms,
                error_message,
            ),
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise RuntimeError(f"[orchestrator] Failed to write audit_log: {exc}") from exc
    finally:
        conn.close()


# ============================================================
# Node shells
#
# Each node follows the same contract:
#   1. Ensure task_state row exists / bump step_index.
#   2. Do its work (deterministic tool OR Qwen reasoning call —
#      the actual Qwen invocation lives in agents/*.py, imported
#      here; this file only shells out to it and manages state).
#   3. Persist via write_audit_log + update_task_state.
#   4. Return the mutated AgentState back to the graph.
#
# Node bodies below are intentionally thin orchestration shells:
# they own state transitions, timing, and persistence. The actual
# Qwen prompt construction and business logic will be implemented
# in agents/intake_parser_agent.py, agents/risk_fraud_agent.py,
# agents/negotiation_agent.py, and agents/pricing_agent.py, and
# imported into these shells in the next build step.
# ============================================================

def intake_parser_node(state: AgentState) -> AgentState:
    """Shell for the intake_parser node — RFQ text -> structured line items (Qwen)."""
    node_name = "intake_parser"
    started_at = time.perf_counter()
    state["step_index"] = state.get("step_index", 0) + 1
    state["current_node"] = node_name
    ensure_task_exists(state)

    input_payload = {"rfq_raw_text": state.get("rfq_raw_text", "")}
    try:
        from agents.intake_parser import run_intake_parser  # local import: avoids circular import at module load

        result = run_intake_parser(state)
        state["extracted_items"] = result.get("extracted_items", [])
        state["client_name"] = result.get("client_name", state.get("client_name"))
        state["client_email"] = result.get("client_email", state.get("client_email"))
        state["confidence_score"] = result.get("confidence_score")
        state["reasoning_summary"] = result.get("reasoning_summary")
        state["status"] = "running"
        error_message = None
    except Exception as exc:  # noqa: BLE001 — must never crash the graph
        state["status"] = "error"
        state["error_message"] = str(exc)
        error_message = str(exc)

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    output_payload = {
        "extracted_items": state.get("extracted_items"),
        "confidence_score": state.get("confidence_score"),
    }
    write_audit_log(
        state, node_name, "IntakeParserAgent", input_payload, output_payload,
        latency_ms, next_node="pricing_calculator", error_message=error_message,
    )
    update_task_state(state)
    return state


def pricing_calculator_node(state: AgentState) -> AgentState:
    """
    Deterministic tool node — NOT an LLM call. Reads pricing_rules and
    computes quote_amount / margin_pct directly from extracted_items.
    """
    node_name = "pricing_calculator"
    started_at = time.perf_counter()
    state["step_index"] = state.get("step_index", 0) + 1
    state["current_node"] = node_name

    input_payload = {"extracted_items": state.get("extracted_items", [])}
    error_message = None
    try:
        from agents.pricing_agent import calculate_quote  # deterministic tool, no Qwen call

        quote_amount, margin_pct = calculate_quote(state.get("extracted_items", []), db_path=DB_PATH)
        state["quote_amount"] = quote_amount
        state["margin_pct"] = margin_pct
        state["status"] = "running"
    except Exception as exc:  # noqa: BLE001
        state["status"] = "error"
        state["error_message"] = str(exc)
        error_message = str(exc)

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    output_payload = {"quote_amount": state.get("quote_amount"), "margin_pct": state.get("margin_pct")}
    write_audit_log(
        state, node_name, "PricingCalculatorTool", input_payload, output_payload,
        latency_ms, next_node="risk_fraud_assessor", tool_call_name="calculate_quote",
        error_message=error_message,
    )
    update_task_state(state)
    return state


def risk_fraud_assessor_node(state: AgentState) -> AgentState:
    """Shell for the risk_fraud_assessor node — Qwen scores fraud/risk (0-1)."""
    node_name = "risk_fraud_assessor"
    started_at = time.perf_counter()
    state["step_index"] = state.get("step_index", 0) + 1
    state["current_node"] = node_name

    input_payload = {
        "client_email": state.get("client_email"),
        "quote_amount": state.get("quote_amount"),
    }
    error_message = None
    try:
        from agents.risk_fraud_agent import assess_risk

        result = assess_risk(state, db_path=DB_PATH)
        state["risk_score"] = result.get("risk_score")
        state["reasoning_summary"] = result.get("reasoning_summary")
        state["status"] = "running"
    except Exception as exc:  # noqa: BLE001
        state["status"] = "error"
        state["error_message"] = str(exc)
        error_message = str(exc)

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    next_node = "human_approval_gate" if _is_high_risk(state) else "negotiation_agent"
    if next_node == "human_approval_gate":
        state["status"] = "pending_approval"
    output_payload = {"risk_score": state.get("risk_score")}
    write_audit_log(
        state, node_name, "RiskFraudAssessorAgent", input_payload, output_payload,
        latency_ms, next_node=next_node, error_message=error_message,
    )
    update_task_state(state)
    return state


def negotiation_agent_node(state: AgentState) -> AgentState:
    """Shell for the negotiation_agent node — Qwen proposes terms + confidence_score."""
    node_name = "negotiation_agent"
    started_at = time.perf_counter()
    state["step_index"] = state.get("step_index", 0) + 1
    state["current_node"] = node_name
    state["negotiation_rounds"] = state.get("negotiation_rounds", 0) + 1

    input_payload = {
        "quote_amount": state.get("quote_amount"),
        "margin_pct": state.get("margin_pct"),
        "negotiation_rounds": state.get("negotiation_rounds"),
    }
    error_message = None
    try:
        from agents.negotiation_agent import run_negotiation

        result = run_negotiation(state)
        state["confidence_score"] = result.get("confidence_score")
        state["reasoning_summary"] = result.get("reasoning_summary")
        state["quote_amount"] = result.get("quote_amount", state.get("quote_amount"))
        state["status"] = "running"
    except Exception as exc:  # noqa: BLE001
        state["status"] = "error"
        state["error_message"] = str(exc)
        error_message = str(exc)

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    next_node = "pdf_quote_generator" if _is_confident_enough(state) else "human_approval_gate"
    output_payload = {"confidence_score": state.get("confidence_score"), "quote_amount": state.get("quote_amount")}
    write_audit_log(
        state, node_name, "NegotiationAgent", input_payload, output_payload,
        latency_ms, next_node=next_node, error_message=error_message,
    )
    if next_node == "human_approval_gate":
        state["status"] = "pending_approval"
    update_task_state(state)
    return state


def human_approval_gate_node(state: AgentState) -> AgentState:
    """
    Interrupt checkpoint — LangGraph pauses execution here via
    interrupt_before. The dashboard writes to human_approvals and
    resumes this thread's checkpoint with human_decision populated.
    """
    node_name = "human_approval_gate"
    started_at = time.perf_counter()
    state["step_index"] = state.get("step_index", 0) + 1
    state["current_node"] = node_name

    input_payload = {
        "risk_score": state.get("risk_score"),
        "confidence_score": state.get("confidence_score"),
        "quote_amount": state.get("quote_amount"),
    }
    decision = state.get("human_decision")
    error_message = None

    if decision is None:
        # Graph is paused here awaiting a resumed invocation with
        # human_decision set by the dashboard's Approve/Reject action.
        state["status"] = "pending_approval"
        next_node = None
    elif decision == "approved":
        state["status"] = "approved"
        next_node = "pdf_quote_generator"
    elif decision == "rejected":
        state["status"] = "rejected"
        next_node = "end"
    else:
        state["status"] = "error"
        state["error_message"] = f"Invalid human_decision value: {decision}"
        error_message = state["error_message"]
        next_node = "end"

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    output_payload = {"human_decision": decision, "reviewer": state.get("human_reviewer")}
    write_audit_log(
        state, node_name, "HumanApprovalGate", input_payload, output_payload,
        latency_ms, next_node=next_node, error_message=error_message,
    )
    update_task_state(state)
    return state


def pdf_quote_generator_node(state: AgentState) -> AgentState:
    """Shell for pdf_quote_generator — renders the final quote PDF and uploads to OSS."""
    node_name = "pdf_quote_generator"
    started_at = time.perf_counter()
    state["step_index"] = state.get("step_index", 0) + 1
    state["current_node"] = node_name

    input_payload = {"quote_amount": state.get("quote_amount"), "client_name": state.get("client_name")}
    error_message = None
    try:
        from agents.pdf_generation_agent import generate_and_upload_quote

        oss_url = generate_and_upload_quote(state)
        state["pdf_oss_url"] = oss_url
        state["status"] = "finalized"
    except Exception as exc:  # noqa: BLE001
        state["status"] = "error"
        state["error_message"] = str(exc)
        error_message = str(exc)

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    output_payload = {"pdf_oss_url": state.get("pdf_oss_url")}
    write_audit_log(
        state, node_name, "PDFQuoteGeneratorAgent", input_payload, output_payload,
        latency_ms, next_node="end", error_message=error_message,
    )
    update_task_state(state)
    return state


# ============================================================
# Conditional edge routers
# ============================================================

def _is_high_risk(state: AgentState) -> bool:
    """True when risk_score meets/exceeds the configured risk_threshold."""
    risk_score = state.get("risk_score")
    return risk_score is not None and risk_score >= RISK_THRESHOLD


def _is_confident_enough(state: AgentState) -> bool:
    """True when confidence_score meets/exceeds the configured confidence_threshold."""
    confidence_score = state.get("confidence_score")
    return confidence_score is not None and confidence_score >= CONFIDENCE_THRESHOLD


def route_after_risk_assessment(state: AgentState) -> str:
    """
    Conditional edge after risk_fraud_assessor:
        risk_score >= risk_threshold -> human_approval_gate
        risk_score <  risk_threshold -> negotiation_agent
    """
    if state.get("status") == "error":
        return "error_end"
    return "human_approval_gate" if _is_high_risk(state) else "negotiation_agent"


def route_after_negotiation(state: AgentState) -> str:
    """
    Conditional edge after negotiation_agent:
        confidence_score >= confidence_threshold -> pdf_quote_generator
        confidence_score <  confidence_threshold -> human_approval_gate
    """
    if state.get("status") == "error":
        return "error_end"
    return "pdf_quote_generator" if _is_confident_enough(state) else "human_approval_gate"


def route_after_human_approval(state: AgentState) -> str:
    """
    Conditional edge after human_approval_gate:
        approved -> pdf_quote_generator
        rejected -> end
        (pending / None should never reach here — interrupt_before pauses first)
    """
    if state.get("status") == "error":
        return "error_end"
    if state.get("status") == "approved":
        return "pdf_quote_generator"
    return "end"


def route_after_pdf_generation(state: AgentState) -> str:
    """Conditional edge after pdf_quote_generator: always terminal."""
    return "error_end" if state.get("status") == "error" else "end"


# ============================================================
# Graph construction
# ============================================================

def build_graph() -> StateGraph:
    """
    Construct the QuoteFlow AI v3.0 StateGraph exactly per the blueprint's
    node map, with interrupt_before=['human_approval_gate'] for native
    human-in-the-loop pausing.
    """
    graph = StateGraph(AgentState)

    graph.add_node("intake_parser", intake_parser_node)
    graph.add_node("pricing_calculator", pricing_calculator_node)
    graph.add_node("risk_fraud_assessor", risk_fraud_assessor_node)
    graph.add_node("negotiation_agent", negotiation_agent_node)
    graph.add_node("human_approval_gate", human_approval_gate_node)
    graph.add_node("pdf_quote_generator", pdf_quote_generator_node)

    graph.add_edge(START, "intake_parser")
    graph.add_edge("intake_parser", "pricing_calculator")
    graph.add_edge("pricing_calculator", "risk_fraud_assessor")

    graph.add_conditional_edges(
        "risk_fraud_assessor",
        route_after_risk_assessment,
        {
            "negotiation_agent": "negotiation_agent",
            "human_approval_gate": "human_approval_gate",
            "error_end": END,
        },
    )

    graph.add_conditional_edges(
        "negotiation_agent",
        route_after_negotiation,
        {
            "pdf_quote_generator": "pdf_quote_generator",
            "human_approval_gate": "human_approval_gate",
            "error_end": END,
        },
    )

    graph.add_conditional_edges(
        "human_approval_gate",
        route_after_human_approval,
        {
            "pdf_quote_generator": "pdf_quote_generator",
            "end": END,
            "error_end": END,
        },
    )

    graph.add_conditional_edges(
        "pdf_quote_generator",
        route_after_pdf_generation,
        {
            "end": END,
            "error_end": END,
        },
    )

    return graph


def compile_orchestrator():
    """
    Compile the StateGraph with a SQLite-backed checkpointer, enabling
    native pause/resume of the human_approval_gate node across process
    restarts. Returns the compiled, runnable graph.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = build_graph()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=SETTINGS["nodes"]["interrupt_before"],
    )
    return compiled


def create_new_task(rfq_raw_text: str, client_name: Optional[str] = None,
                     client_email: Optional[str] = None) -> AgentState:
    """
    Factory for a fresh AgentState / thread_id pair, ready to be invoked
    against the compiled orchestrator graph.
    """
    task_id = str(uuid.uuid4())
    thread_id = f"thread-{task_id}"
    return AgentState(
        task_id=task_id,
        thread_id=thread_id,
        step_index=0,
        client_name=client_name,
        client_email=client_email,
        rfq_raw_text=rfq_raw_text,
        extracted_items=[],
        negotiation_rounds=0,
        current_node="intake_parser",
        status="running",
        error_message=None,
    )


# Module-level compiled instance, imported by backend/main.py (FastAPI
# entrypoint) and dashboard.py's resume-thread action.
orchestrator_app = compile_orchestrator()