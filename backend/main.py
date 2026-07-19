"""
backend/main.py

QuoteFlow AI v3.0 — FastAPI Serverless Entrypoint

This is the thin HTTP surface over the LangGraph orchestrator. It does NOT
contain any business logic, reasoning, or routing decisions itself — every
request either:
    (a) creates a new AgentState and invokes the compiled StateGraph, or
    (b) resumes a paused thread at human_approval_gate via LangGraph's
        native checkpoint/update_state mechanism, or
    (c) reads directly from the SQLite task_state / audit_log tables
        (the same tables dashboard.py polls) for status/history queries.

Deployed as the handler behind Alibaba Cloud Function Compute (see
alibaba_services.py for the deployment-proof integration); runs equally
well locally via `uvicorn backend.main:app --reload`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional


import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.orchestrator import (
    AgentState,
    DB_PATH,
    create_new_task,
    orchestrator_app,
)

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

_APP_CFG = _SETTINGS["app"]


# ============================================================
# Database helper (read-only queries for status/history endpoints)
# ============================================================

def _get_connection() -> sqlite3.Connection:
    """Open a short-lived read connection to the shared SQLite database."""
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


# ============================================================
# Lifespan — verify DB is initialized on startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify the SQLite database and schema are reachable before serving traffic."""
    if not os.path.exists(DB_PATH):
        raise RuntimeError(
            f"[main] SQLite database not found at {DB_PATH}. "
            "Run `sqlite3 db/quoteflow.db < db/schema.sql` before starting the server."
        )
    conn = _get_connection()
    try:
        conn.execute("SELECT COUNT(*) FROM task_state;")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            f"[main] task_state table is missing or schema is not applied: {exc}"
        ) from exc
    finally:
        conn.close()
    yield


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="QuoteFlow AI v3.0",
    description=(
        "Autonomous B2B quote-generation pipeline orchestrated with LangGraph "
        "and reasoned by Qwen2.5, deployed serverless on Alibaba Cloud."
    ),
    version=_APP_CFG["version"],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened at the Alibaba Cloud API Gateway layer in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic request/response schemas
# ============================================================

class RFQSubmissionRequest(BaseModel):
    """Payload for submitting a new RFQ into the LangGraph pipeline."""

    rfq_raw_text: str = Field(..., min_length=10, description="Raw inbound RFQ email/document text")
    client_name: Optional[str] = Field(None, description="Optional pre-known client name")
    client_email: Optional[EmailStr] = Field(None, description="Optional pre-known client email")

    @field_validator("rfq_raw_text")
    @classmethod
    def _rfq_must_have_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rfq_raw_text cannot be blank or whitespace-only")
        return value.strip()


class RFQSubmissionResponse(BaseModel):
    task_id: str
    thread_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    thread_id: str
    client_name: Optional[str]
    client_email: Optional[str]
    current_node: str
    status: str
    latest_confidence: Optional[float]
    latest_risk_score: Optional[float]
    quote_amount: Optional[float]
    margin_pct: Optional[float]
    pdf_oss_url: Optional[str]
    created_at: str
    updated_at: str


class AuditLogEntry(BaseModel):
    log_id: int
    task_id: str
    thread_id: str
    node_name: str
    agent_name: str
    step_index: int
    reasoning_summary: Optional[str]
    confidence_score: Optional[float]
    risk_score: Optional[float]
    next_node: Optional[str]
    latency_ms: Optional[int]
    error_message: Optional[str]
    created_at: str


class HumanApprovalRequest(BaseModel):
    """Payload for resolving a human_approval_gate interrupt from the dashboard."""

    decision: str = Field(..., description="'approved' or 'rejected'")
    decided_by: str = Field(..., min_length=1, description="Human reviewer identifier")
    reason: Optional[str] = Field(None, description="Optional justification for the decision")

    @field_validator("decision")
    @classmethod
    def _validate_decision(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ("approved", "rejected"):
            raise ValueError("decision must be exactly 'approved' or 'rejected'")
        return normalized


class HumanApprovalResponse(BaseModel):
    task_id: str
    decision: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str


# ============================================================
# Background execution helper
# ============================================================

def _run_graph_to_completion_or_pause(initial_state: AgentState) -> None:
    """
    Invoke the compiled LangGraph orchestrator for a fresh thread. Execution
    proceeds through intake_parser -> pricing_calculator -> risk_fraud_assessor
    -> (negotiation_agent | human_approval_gate) and either finalizes or pauses
    at human_approval_gate (interrupt_before), all state changes being written
    to SQLite by each node shell as they execute.
    """
    thread_config = {"configurable": {"thread_id": initial_state["thread_id"]}}
    try:
        orchestrator_app.invoke(initial_state, config=thread_config)
    except Exception as exc:  # noqa: BLE001 — node shells already persist their own errors;
        # this catch-all guards against a failure occurring outside any single
        # node's try/except (e.g. graph engine itself), so it's never lost silently.
        conn = _get_connection()
        try:
            conn.execute(
                """
                UPDATE task_state
                SET status = 'error', updated_at = datetime('now')
                WHERE task_id = ?
                """,
                (initial_state["task_id"],),
            )
            conn.commit()
        finally:
            conn.close()
        print(f"[main] Unhandled orchestrator error for task {initial_state['task_id']}: {exc}")


def _resume_graph_after_approval(thread_id: str, human_decision: str, reviewer: str) -> None:
    """
    Resume a paused human_approval_gate thread by updating the checkpointed
    state with the human's decision, then re-invoking the graph so LangGraph's
    conditional edge (route_after_human_approval) can proceed.
    """
    thread_config = {"configurable": {"thread_id": thread_id}}
    try:
        orchestrator_app.update_state(
            thread_config,
            {"human_decision": human_decision, "human_reviewer": reviewer},
        )
        orchestrator_app.invoke(None, config=thread_config)
    except Exception as exc:  # noqa: BLE001
        conn = _get_connection()
        try:
            conn.execute(
                """
                UPDATE task_state
                SET status = 'error', updated_at = datetime('now')
                WHERE thread_id = ?
                """,
                (thread_id,),
            )
            conn.commit()
        finally:
            conn.close()
        print(f"[main] Unhandled resume error for thread {thread_id}: {exc}")


# ============================================================
# Routes
# ============================================================

@app.get("/", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Lightweight liveness/readiness probe for Alibaba Cloud Function Compute."""
    return HealthResponse(
        status="ok",
        app_name=_APP_CFG["name"],
        version=_APP_CFG["version"],
        environment=_APP_CFG["environment"],
    )


@app.post(
    "/api/v1/rfq/submit",
    response_model=RFQSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["RFQ Pipeline"],
)
async def submit_rfq(
    payload: RFQSubmissionRequest,
    background_tasks: BackgroundTasks,
) -> RFQSubmissionResponse:
    """
    Submit a new RFQ into the QuoteFlow AI pipeline.

    Creates a fresh AgentState + LangGraph thread and schedules graph
    execution as a background task so the HTTP request returns immediately
    with a task_id the client can poll via GET /api/v1/rfq/{task_id}/status.
    """
    initial_state = create_new_task(
        rfq_raw_text=payload.rfq_raw_text,
        client_name=payload.client_name,
        client_email=payload.client_email,
    )

    background_tasks.add_task(_run_graph_to_completion_or_pause, initial_state)

    return RFQSubmissionResponse(
        task_id=initial_state["task_id"],
        thread_id=initial_state["thread_id"],
        status="running",
        message="RFQ accepted and pipeline execution started.",
    )


@app.get(
    "/api/v1/rfq/{task_id}/status",
    response_model=TaskStatusResponse,
    tags=["RFQ Pipeline"],
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Fetch the live status of a task directly from task_state — the same
    row the Streamlit dashboard polls for real-time node visibility.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM task_state WHERE task_id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No task found with task_id={task_id}")

    return TaskStatusResponse(**_row_to_dict(row))


@app.get(
    "/api/v1/rfq/{task_id}/audit-log",
    response_model=list[AuditLogEntry],
    tags=["RFQ Pipeline"],
)
async def get_task_audit_log(task_id: str) -> list[AuditLogEntry]:
    """
    Return the full ordered audit trail for a task — every LangGraph node
    execution, confidence/risk score, and routing decision, oldest first.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT log_id, task_id, thread_id, node_name, agent_name, step_index,
                   reasoning_summary, confidence_score, risk_score, next_node,
                   latency_ms, error_message, created_at
            FROM audit_log
            WHERE task_id = ?
            ORDER BY step_index ASC, log_id ASC
            """,
            (task_id,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No audit log entries found for task_id={task_id}",
        )

    return [AuditLogEntry(**_row_to_dict(row)) for row in rows]


@app.get("/api/v1/rfq/{task_id}/quote-pdf", tags=["RFQ Pipeline"])
async def get_quote_pdf(task_id: str):
    """
    Unified quote PDF access point — transparently serves the file whether
    it lives locally ('local://...') or on Alibaba Cloud OSS (a presigned
    URL), so the dashboard's "View Quote" link always works identically
    regardless of storage backend.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT pdf_oss_url FROM task_state WHERE task_id = ?", (task_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or not row["pdf_oss_url"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No quote PDF found for task_id={task_id}",
        )

    pdf_reference = row["pdf_oss_url"]

    if pdf_reference.startswith("local://"):
        local_path = pdf_reference.replace("local://", "", 1)
        if not os.path.exists(local_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The local quote PDF file no longer exists on disk.",
            )
        return FileResponse(
            local_path, media_type="application/pdf", filename=os.path.basename(local_path)
        )

    return RedirectResponse(url=pdf_reference)



@app.get(
    "/api/v1/rfq/pending-approval",
    response_model=list[TaskStatusResponse],
    tags=["Human-in-the-Loop"],
)
async def list_pending_approvals() -> list[TaskStatusResponse]:
    """
    List every task currently paused at human_approval_gate, awaiting a
    dashboard reviewer decision. Backs the dashboard's Pending HITL table.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM task_state WHERE status = 'pending_approval' ORDER BY updated_at ASC"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [TaskStatusResponse(**_row_to_dict(row)) for row in rows]


@app.post(
    "/api/v1/rfq/{task_id}/approve",
    response_model=HumanApprovalResponse,
    tags=["Human-in-the-Loop"],
)
async def resolve_human_approval(
    task_id: str,
    payload: HumanApprovalRequest,
    background_tasks: BackgroundTasks,
) -> HumanApprovalResponse:
    """
    Resolve a paused human_approval_gate checkpoint with an Approve/Reject
    decision from the dashboard. Writes the decision to human_approvals,
    then resumes the LangGraph thread via its checkpoint so execution
    continues to pdf_quote_generator (approved) or terminates (rejected).
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT thread_id, status FROM task_state WHERE task_id = ?",
            (task_id,),
        )
        row = cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No task found with task_id={task_id}",
            )

        if row["status"] != "pending_approval":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Task {task_id} is not awaiting approval "
                    f"(current status: '{row['status']}')"
                ),
            )

        thread_id = row["thread_id"]

        conn.execute(
            """
            INSERT INTO human_approvals (task_id, decision, decided_by, reason)
            VALUES (?, ?, ?, ?)
            """,
            (task_id, payload.decision, payload.decided_by, payload.reason),
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while recording approval decision: {exc}",
        ) from exc
    finally:
        conn.close()

    background_tasks.add_task(
        _resume_graph_after_approval, thread_id, payload.decision, payload.decided_by
    )

    return HumanApprovalResponse(
        task_id=task_id,
        decision=payload.decision,
        status="approved" if payload.decision == "approved" else "rejected",
        message=f"Decision recorded and LangGraph thread {thread_id} resumed.",
    )


@app.get(
    "/api/v1/rfq/list",
    response_model=list[TaskStatusResponse],
    tags=["RFQ Pipeline"],
)
async def list_recent_tasks(limit: int = 50) -> list[TaskStatusResponse]:
    """
    List the most recently updated tasks across all statuses, newest first.
    Backs the dashboard's main operations table.
    """
    if limit <= 0 or limit > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 500",
        )

    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM task_state ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [TaskStatusResponse(**_row_to_dict(row)) for row in rows]


# ============================================================
# Global exception handling — never leak raw tracebacks to clients
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):  # noqa: ANN001
    """Catch-all safety net so unexpected errors return a clean JSON 500."""
    from fastapi.responses import JSONResponse

    print(f"[main] Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. This has been logged for review."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=_APP_CFG["environment"] != "production",
    )