"""
dashboard.py

QuoteFlow AI v3.0 — Live Operations Dashboard (Streamlit)

Design intent: a high-end, developer-built B2B SaaS console (Stripe/Retool-grade),
not a default Streamlit demo. All metrics, trend charts, and the live node
visualizer are read directly from SQLite (task_state / audit_log) — the same
tables backend/orchestrator.py writes to on every node transition. Zero
hardcoded series or fake data anywhere in this file.

IMPORTANT RENDERING RULE: every HTML fragment injected via st.markdown(...,
unsafe_allow_html=True) MUST be built as a single-line string with zero
leading whitespace. CommonMark treats 4+ leading spaces as a code block,
which renders BEFORE unsafe_allow_html can interpret the tags — this is
what caused the earlier "raw HTML dumped as text" bug. Never reintroduce
indented multi-line f-strings for HTML in this file.

Actions that mutate pipeline state (submitting a new RFQ, resolving a
human_approval_gate decision) call the FastAPI backend, since only the
running orchestrator process can correctly resume a LangGraph checkpoint.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yaml

# ============================================================
# Configuration
# ============================================================

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "settings.yaml")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _handle:
    _SETTINGS = yaml.safe_load(_handle)

_DB_PATH = os.path.join(_PROJECT_ROOT, _SETTINGS["database"]["sqlite_path"])
_DASHBOARD_CFG = _SETTINGS["dashboard"]
_NODE_ORDER: list[str] = _SETTINGS["nodes"]["order"]
_REFRESH_SECONDS: int = _DASHBOARD_CFG["refresh_interval_seconds"]
_MAX_RECENT_LOGS: int = _DASHBOARD_CFG["max_recent_logs"]

_BACKEND_API_BASE_URL = os.environ.get("QUOTEFLOW_API_BASE_URL", "http://localhost:8000")

_NODE_LABELS: dict[str, str] = {
    "intake_parser": "Intake Parser",
    "pricing_calculator": "Pricing Calculator",
    "risk_fraud_assessor": "Risk & Fraud Assessor",
    "negotiation_agent": "Negotiation Agent",
    "human_approval_gate": "Human Approval Gate",
    "pdf_quote_generator": "PDF Quote Generator",
}

_STATUS_COLORS: dict[str, str] = {
    "running": "#3B82F6",
    "pending_approval": "#F59E0B",
    "approved": "#10B981",
    "rejected": "#EF4444",
    "finalized": "#22C55E",
    "error": "#DC2626",
}

_STATUS_LABELS: dict[str, str] = {
    "running": "Running",
    "pending_approval": "Pending Approval",
    "approved": "Approved",
    "rejected": "Rejected",
    "finalized": "Finalized",
    "error": "Error",
}


# ============================================================
# Page config — must be first Streamlit call
# ============================================================

st.set_page_config(
    page_title=_DASHBOARD_CFG["page_title"],
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS — enterprise SaaS look, no default Streamlit chrome
# ============================================================

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
footer { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stToolbar"] { display: none; }

.stApp {
    background: #0A0E17;
}
.block-container {
    padding-top: 1.75rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background: #0D1220;
    border-right: 1px solid #1E2536;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.qf-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid #1E2536;
    margin-bottom: 1.75rem;
}
.qf-header-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.02em;
}
.qf-header-title span {
    color: #3B82F6;
}
.qf-header-subtitle {
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 500;
    margin-top: 2px;
}
.qf-header-badge {
    font-size: 0.7rem;
    font-weight: 600;
    color: #94A3B8;
    background: #131A2A;
    border: 1px solid #1E2536;
    padding: 5px 12px;
    border-radius: 6px;
    letter-spacing: 0.03em;
    font-family: 'JetBrains Mono', monospace;
}

.qf-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.75rem 0 0.85rem 0;
}

.qf-kpi-card {
    background: #10141F;
    border: 1px solid #1E2536;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    height: 100%;
}
.qf-kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.qf-kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #F1F5F9;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}

.qf-panel {
    background: #10141F;
    border: 1px solid #1E2536;
    border-radius: 10px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.25rem;
}
.qf-panel-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 0.9rem;
}

.qf-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 10px 4px 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #E2E8F0;
    white-space: nowrap;
}
.qf-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.qf-dot.blink {
    animation: qf-pulse 1.4s ease-in-out infinite;
}
@keyframes qf-pulse {
    0%   { opacity: 1;   box-shadow: 0 0 0 0 currentColor; }
    70%  { opacity: 0.55; box-shadow: 0 0 0 4px transparent; }
    100% { opacity: 1;   box-shadow: 0 0 0 0 transparent; }
}

.qf-pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
    padding: 0.25rem 0 0.5rem 0;
}
.qf-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 138px;
    position: relative;
    padding: 0 6px;
}
.qf-node-circle {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #171E2E;
    border: 2px solid #2A3348;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748B;
    margin-bottom: 8px;
    z-index: 1;
    transition: all 0.3s ease;
}
.qf-node.active .qf-node-circle {
    background: #1D2A4A;
    border-color: #3B82F6;
    color: #60A5FA;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.15);
    animation: qf-node-glow 1.6s ease-in-out infinite;
}
.qf-node.done .qf-node-circle {
    background: #14203A;
    border-color: #3B82F6;
    color: #3B82F6;
}
@keyframes qf-node-glow {
    0%, 100% { box-shadow: 0 0 0 4px rgba(59,130,246,0.15); }
    50%      { box-shadow: 0 0 0 7px rgba(59,130,246,0.08); }
}
.qf-node-line {
    position: absolute;
    top: 19px;
    left: 50%;
    width: 100%;
    height: 2px;
    background: #1E2536;
    z-index: 0;
    transition: background 0.3s ease;
}
.qf-node.active .qf-node-line,
.qf-node.done .qf-node-line {
    background: #3B82F6;
}
.qf-node:last-child .qf-node-line { display: none; }
.qf-node-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #64748B;
    text-align: center;
    line-height: 1.25;
}
.qf-node.active .qf-node-label { color: #E2E8F0; }
.qf-node.done .qf-node-label { color: #94A3B8; }
.qf-node-count {
    font-size: 0.6rem;
    color: #475569;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #1E2536;
    border-radius: 8px;
    overflow: hidden;
}

.stButton > button {
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.82rem;
    border: 1px solid #1E2536;
}
.stButton > button[kind="primary"] {
    background: #3B82F6;
    border: none;
}

.qf-divider {
    border: none;
    border-top: 1px solid #1E2536;
    margin: 1.75rem 0;
}

.qf-empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #475569;
    font-size: 0.85rem;
}
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Database access layer (cached, read-only)
# ============================================================

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=_REFRESH_SECONDS, show_spinner=False)
def load_task_summary() -> dict[str, Any]:
    """Aggregate KPI counters across all tasks — zero hardcoded numbers."""
    conn = _get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) AS c FROM task_state").fetchone()["c"]
        by_status = conn.execute(
            "SELECT status, COUNT(*) AS c FROM task_state GROUP BY status"
        ).fetchall()
        avg_confidence = conn.execute(
            "SELECT AVG(latest_confidence) AS v FROM task_state WHERE latest_confidence IS NOT NULL"
        ).fetchone()["v"]
        avg_risk = conn.execute(
            "SELECT AVG(latest_risk_score) AS v FROM task_state WHERE latest_risk_score IS NOT NULL"
        ).fetchone()["v"]
        total_quote_value = conn.execute(
            "SELECT SUM(quote_amount) AS v FROM task_state WHERE status = 'finalized'"
        ).fetchone()["v"]
    finally:
        conn.close()

    status_counts = {row["status"]: row["c"] for row in by_status}
    return {
        "total": total,
        "status_counts": status_counts,
        "avg_confidence": avg_confidence,
        "avg_risk": avg_risk,
        "total_quote_value": total_quote_value or 0.0,
    }


@st.cache_data(ttl=_REFRESH_SECONDS, show_spinner=False)
def load_recent_tasks(limit: int = 100) -> pd.DataFrame:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM task_state ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=_REFRESH_SECONDS, show_spinner=False)
def load_pending_approvals() -> pd.DataFrame:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM task_state WHERE status = 'pending_approval' ORDER BY updated_at ASC"
        )
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=_REFRESH_SECONDS, show_spinner=False)
def load_active_node_counts() -> dict[str, int]:
    """Count of tasks currently sitting at each node — feeds the pipeline visualizer."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT current_node, COUNT(*) AS c
            FROM task_state
            WHERE status IN ('running', 'pending_approval')
            GROUP BY current_node
            """
        )
        rows = cursor.fetchall()
    finally:
        conn.close()
    return {row["current_node"]: row["c"] for row in rows}


@st.cache_data(ttl=_REFRESH_SECONDS, show_spinner=False)
def load_confidence_risk_trend() -> pd.DataFrame:
    """
    Pull confidence_score / risk_score grouped by node_name over time from
    audit_log — feeds the trend chart. Purely derived, no synthetic points.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT log_id, node_name, confidence_score, risk_score, created_at
            FROM audit_log
            WHERE confidence_score IS NOT NULL OR risk_score IS NOT NULL
            ORDER BY log_id ASC
            LIMIT ?
            """,
            (_MAX_RECENT_LOGS,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


@st.cache_data(ttl=_REFRESH_SECONDS, show_spinner=False)
def load_audit_log_for_task(task_id: str) -> pd.DataFrame:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT step_index, node_name, agent_name, reasoning_summary,
                   confidence_score, risk_score, next_node, latency_ms,
                   error_message, created_at
            FROM audit_log
            WHERE task_id = ?
            ORDER BY step_index ASC
            """,
            (task_id,),
        )
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return pd.DataFrame(rows)

@st.cache_data(ttl=_REFRESH_SECONDS, show_spinner=False)
def load_last_completed_trace() -> dict[str, Any]:
    """
    Fetch the most recently updated task's full node visit history from
    audit_log. Used to show a 'Last Run' trace on the pipeline visualizer
    when no task is currently in-flight, so the visualizer never looks
    empty right after a task finishes.
    """
    conn = _get_connection()
    try:
        task_row = conn.execute(
            "SELECT task_id, client_name, status FROM task_state ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if task_row is None:
            return {}

        visited_rows = conn.execute(
            "SELECT DISTINCT node_name FROM audit_log WHERE task_id = ?",
            (task_row["task_id"],),
        ).fetchall()
        visited_nodes = {row["node_name"] for row in visited_rows}

        return {
            "task_id": task_row["task_id"],
            "client_name": task_row["client_name"] or "Unknown Client",
            "status": task_row["status"],
            "visited_nodes": visited_nodes,
        }
    finally:
        conn.close()


def clear_all_caches() -> None:
    load_task_summary.clear()
    load_recent_tasks.clear()
    load_pending_approvals.clear()
    load_active_node_counts.clear()
    load_confidence_risk_trend.clear()
    load_audit_log_for_task.clear()
    load_last_completed_trace.clear()


# ============================================================
# Backend API helpers (state-mutating actions only)
# ============================================================

def submit_rfq_to_backend(rfq_raw_text: str, client_name: str, client_email: str) -> dict[str, Any]:
    """POST a new RFQ to FastAPI so the orchestrator starts a fresh LangGraph thread."""
    payload = {
        "rfq_raw_text": rfq_raw_text,
        "client_name": client_name or None,
        "client_email": client_email or None,
    }
    response = requests.post(
        f"{_BACKEND_API_BASE_URL}/api/v1/rfq/submit", json=payload, timeout=15
    )
    response.raise_for_status()
    return response.json()


def resolve_approval_via_backend(task_id: str, decision: str, decided_by: str, reason: str) -> dict[str, Any]:
    """POST an Approve/Reject decision so the backend resumes the paused LangGraph checkpoint."""
    payload = {"decision": decision, "decided_by": decided_by, "reason": reason or None}
    response = requests.post(
        f"{_BACKEND_API_BASE_URL}/api/v1/rfq/{task_id}/approve", json=payload, timeout=15
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# UI rendering — small components
#
# RULE: every string passed to st.markdown(..., unsafe_allow_html=True)
# is built as ONE continuous line (or joined fragments with zero leading
# whitespace) — never an indented multi-line f-string. Indented lines
# get parsed as a Markdown code block before the HTML is ever rendered.
# ============================================================

def render_status_badge(status_value: str) -> str:
    color = _STATUS_COLORS.get(status_value, "#64748B")
    label = _STATUS_LABELS.get(status_value, status_value.title())
    blink_class = "blink" if status_value in ("running", "pending_approval") else ""
    return (
        f'<span class="qf-badge"><span class="qf-dot {blink_class}" '
        f'style="background:{color}; color:{color};"></span>{label}</span>'
    )


def render_header() -> None:
    html = (
        '<div class="qf-header">'
        '<div>'
        '<div class="qf-header-title">Quote<span>Flow</span> AI</div>'
        f'<div class="qf-header-subtitle">Live Autonomous Quoting Operations · v{_SETTINGS["app"]["version"]}</div>'
        '</div>'
        '<div class="qf-header-badge">LANGGRAPH ORCHESTRATOR · QWEN2.5 REASONING · ALIBABA CLOUD</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_grid(summary: dict[str, Any]) -> None:
    status_counts = summary["status_counts"]
    running = status_counts.get("running", 0) + status_counts.get("pending_approval", 0)
    finalized = status_counts.get("finalized", 0)
    rejected = status_counts.get("rejected", 0)
    errors = status_counts.get("error", 0)
    avg_conf = summary["avg_confidence"]
    avg_risk = summary["avg_risk"]

    cols = st.columns(6)
    kpis = [
        ("Total RFQs", f"{summary['total']:,}"),
        ("Active Threads", f"{running:,}"),
        ("Finalized", f"{finalized:,}"),
        ("Rejected", f"{rejected:,}"),
        ("Avg Confidence", f"{avg_conf:.0%}" if avg_conf is not None else "—"),
        ("Avg Risk Score", f"{avg_risk:.0%}" if avg_risk is not None else "—"),
    ]
    for col, (label, value) in zip(cols, kpis):
        with col:
            card_html = (
                f'<div class="qf-kpi-card"><div class="qf-kpi-label">{label}</div>'
                f'<div class="qf-kpi-value">{value}</div></div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    if errors > 0:
        alert_html = (
            f'<div style="margin-top:0.6rem;">{render_status_badge("error")} '
            f'<span style="color:#94A3B8; font-size:0.8rem; margin-left:6px;">'
            f'{errors} task(s) require attention</span></div>'
        )
        st.markdown(alert_html, unsafe_allow_html=True)


def render_node_pipeline(active_node_counts: dict[str, int]) -> None:
    st.markdown(
        '<div class="qf-panel"><div class="qf-panel-title">Agent Routing Visualizer</div></div>',
        unsafe_allow_html=True,
    )

    total_active = sum(active_node_counts.values())
    last_trace: dict[str, Any] = {}

    if total_active == 0:
        last_trace = load_last_completed_trace()

    visited_nodes = last_trace.get("visited_nodes", set())
    furthest_index = -1
    for idx, node_name in enumerate(_NODE_ORDER):
        if active_node_counts.get(node_name, 0) > 0:
            furthest_index = idx

    node_fragments = []
    for idx, node_name in enumerate(_NODE_ORDER):
        count = active_node_counts.get(node_name, 0)
        is_active = count > 0
        is_done_live = (not is_active) and (idx < furthest_index)
        is_done_trace = (not is_active) and (not total_active) and (node_name in visited_nodes)
        css_class = "active" if is_active else ("done" if (is_done_live or is_done_trace) else "")
        label = _NODE_LABELS.get(node_name, node_name)
        circle_content = "✓" if (is_done_live or is_done_trace) else (str(count) if count > 0 else "·")
        node_fragments.append(
            f'<div class="qf-node {css_class}"><div class="qf-node-line"></div>'
            f'<div class="qf-node-circle">{circle_content}</div>'
            f'<div class="qf-node-label">{label}</div>'
            f'<div class="qf-node-count">{count} in-flight</div></div>'
        )

    pipeline_html = (
        '<div class="qf-panel" style="margin-top:-1rem;"><div class="qf-pipeline">'
        + "".join(node_fragments)
        + "</div></div>"
    )
    st.markdown(pipeline_html, unsafe_allow_html=True)

    if total_active == 0 and last_trace:
        status_label = _STATUS_LABELS.get(last_trace["status"], last_trace["status"])
        caption_html = (
            f'<div style="text-align:center; color:#64748B; font-size:0.78rem; margin-top:0.4rem;">'
            f'Last Run: <span style="color:#94A3B8; font-weight:600;">{last_trace["client_name"]}</span> '
            f'· <span style="color:#94A3B8;">{status_label}</span></div>'
        )
        st.markdown(caption_html, unsafe_allow_html=True)
    elif total_active == 0:
        st.markdown(
            '<div class="qf-empty-state">No active threads right now — submit an RFQ to see it move through the pipeline.</div>',
            unsafe_allow_html=True,
        )


def render_trend_charts(trend_df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="qf-panel"><div class="qf-panel-title">Confidence &amp; Risk Trend by Node</div></div>',
        unsafe_allow_html=True,
    )

    if trend_df.empty:
        st.markdown(
            '<div class="qf-empty-state">No scored node executions yet — trend data populates as RFQs run through risk and negotiation nodes.</div>',
            unsafe_allow_html=True,
        )
        return

    fig = go.Figure()

    conf_df = trend_df.dropna(subset=["confidence_score"])
    if not conf_df.empty:
        fig.add_trace(
            go.Scatter(
                x=conf_df["created_at"],
                y=conf_df["confidence_score"],
                mode="markers+lines",
                name="Confidence Score",
                line=dict(color="#3B82F6", width=2),
                marker=dict(size=6, color="#3B82F6"),
                hovertemplate="Node: %{text}<br>Confidence: %{y:.2f}<extra></extra>",
                text=conf_df["node_name"],
            )
        )

    risk_df = trend_df.dropna(subset=["risk_score"])
    if not risk_df.empty:
        fig.add_trace(
            go.Scatter(
                x=risk_df["created_at"],
                y=risk_df["risk_score"],
                mode="markers+lines",
                name="Risk Score",
                line=dict(color="#F59E0B", width=2),
                marker=dict(size=6, color="#F59E0B"),
                hovertemplate="Node: %{text}<br>Risk: %{y:.2f}<extra></extra>",
                text=risk_df["node_name"],
            )
        )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94A3B8", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1E2536", showline=False, title=None),
        yaxis=dict(gridcolor="#1E2536", showline=False, range=[0, 1], title=None),
        hoverlabel=dict(bgcolor="#10141F", font_size=12, font_family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_pending_approvals(pending_df: pd.DataFrame) -> None:
    header_html = (
        '<div class="qf-panel"><div class="qf-panel-title">Pending Human Approvals '
        f'<span style="color:#F59E0B; font-weight:700;">({len(pending_df)})</span></div></div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    if pending_df.empty:
        st.markdown(
            '<div class="qf-empty-state">Nothing awaiting review — all in-flight quotes are within automated thresholds.</div>',
            unsafe_allow_html=True,
        )
        return

    for _, task in pending_df.iterrows():
        info_col, action_col = st.columns([4, 2])
        with info_col:
            confidence_text = f"{task['latest_confidence']:.0%}" if pd.notna(task['latest_confidence']) else "—"
            risk_text = f"{task['latest_risk_score']:.0%}" if pd.notna(task['latest_risk_score']) else "—"
            quote_text = f"${task['quote_amount']:.2f}" if pd.notna(task['quote_amount']) else "—"
            row_html = (
                '<div style="padding:0.6rem 0;">'
                f'<div style="font-weight:700; color:#E2E8F0; font-size:0.9rem;">{task["client_name"] or "Unknown Client"} '
                f'<span style="color:#475569; font-weight:500; font-size:0.75rem; margin-left:8px;">{task["task_id"][:8]}…</span></div>'
                f'<div style="color:#64748B; font-size:0.78rem; margin-top:2px;">Quote: '
                f'<span style="color:#94A3B8; font-family:\'JetBrains Mono\',monospace;">{quote_text}</span> · '
                f'Confidence: {confidence_text} · Risk: {risk_text}</div></div>'
            )
            st.markdown(row_html, unsafe_allow_html=True)
        with action_col:
            task_id = task["task_id"]
            if st.button("Approve", key=f"approve_{task_id}", type="primary", use_container_width=True):
                try:
                    resolve_approval_via_backend(task_id, "approved", "dashboard_operator", "")
                    clear_all_caches()
                    st.success(f"Approved {task_id[:8]}… — resuming pipeline.")
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Failed to reach backend: {exc}")
            if st.button("Reject", key=f"reject_{task_id}", use_container_width=True):
                try:
                    resolve_approval_via_backend(task_id, "rejected", "dashboard_operator", "")
                    clear_all_caches()
                    st.warning(f"Rejected {task_id[:8]}….")
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Failed to reach backend: {exc}")
        st.markdown('<hr class="qf-divider" style="margin:0.5rem 0;">', unsafe_allow_html=True)

def render_recent_tasks_table(tasks_df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="qf-panel"><div class="qf-panel-title">Recent RFQ Activity</div></div>',
        unsafe_allow_html=True,
    )

    if tasks_df.empty:
        st.markdown('<div class="qf-empty-state">No RFQs submitted yet.</div>', unsafe_allow_html=True)
        return

    display_df = tasks_df.copy()
    display_df["Client"] = display_df["client_name"].fillna("Unknown")
    display_df["Node"] = display_df["current_node"].map(lambda n: _NODE_LABELS.get(n, n))
    display_df["Status"] = display_df["status"].map(lambda s: _STATUS_LABELS.get(s, s))
    display_df["Quote"] = display_df["quote_amount"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
    display_df["Confidence"] = display_df["latest_confidence"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
    display_df["Risk"] = display_df["latest_risk_score"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
    display_df["Updated"] = pd.to_datetime(display_df["updated_at"], format="mixed",utc=True).dt.strftime("%Y-%m-%d %H:%M")
    display_df["Task ID"] = display_df["task_id"].str[:8] + "…"
    display_df["View Quote"] = display_df.apply(
        lambda row: f"{_BACKEND_API_BASE_URL}/api/v1/rfq/{row['task_id']}/quote-pdf"
        if pd.notna(row["pdf_oss_url"]) else None,
        axis=1,
    )

    st.dataframe(
        display_df[["Task ID", "Client", "Node", "Status", "Quote", "Confidence", "Risk", "View Quote", "Updated"]],
        use_container_width=True,
        hide_index=True,
        height=min(420, 44 + 38 * len(display_df)),
        column_config={
            "View Quote": st.column_config.LinkColumn(
                "View Quote", display_text="📄 View Quote"
            ),
        },
    )

def render_task_inspector(tasks_df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="qf-panel"><div class="qf-panel-title">Task Inspector — Full Audit Trail</div></div>',
        unsafe_allow_html=True,
    )

    if tasks_df.empty:
        st.markdown('<div class="qf-empty-state">No tasks to inspect yet.</div>', unsafe_allow_html=True)
        return

    task_options = {
        f"{row['client_name'] or 'Unknown'} — {row['task_id'][:8]}… ({_STATUS_LABELS.get(row['status'], row['status'])})": row["task_id"]
        for _, row in tasks_df.iterrows()
    }
    selected_label = st.selectbox("Select a task to inspect", options=list(task_options.keys()))
    selected_task_id = task_options[selected_label]

    audit_df = load_audit_log_for_task(selected_task_id)
    if audit_df.empty:
        st.markdown('<div class="qf-empty-state">No audit log entries recorded for this task.</div>', unsafe_allow_html=True)
        return

    display_audit = audit_df.copy()
    display_audit["Node"] = display_audit["node_name"].map(lambda n: _NODE_LABELS.get(n, n))
    display_audit["Confidence"] = display_audit["confidence_score"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
    display_audit["Risk"] = display_audit["risk_score"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
    display_audit["Latency"] = display_audit["latency_ms"].map(lambda v: f"{v:,} ms" if pd.notna(v) else "—")

    st.dataframe(
        display_audit[["step_index", "Node", "agent_name", "Confidence", "Risk", "next_node", "Latency", "created_at"]].rename(
            columns={"step_index": "Step", "agent_name": "Agent", "next_node": "Routed To", "created_at": "Timestamp"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    for _, log_row in display_audit.iterrows():
        if log_row.get("reasoning_summary"):
            node_label = _NODE_LABELS.get(log_row["node_name"], log_row["node_name"])
            note_html = (
                '<div style="background:#0D1220; border:1px solid #1E2536; border-radius:8px; '
                'padding:0.75rem 1rem; margin:0.5rem 0; font-size:0.8rem; color:#94A3B8;">'
                f'<span style="color:#60A5FA; font-weight:600;">Step {log_row["step_index"]} · {node_label}:</span> '
                f'{log_row["reasoning_summary"]}</div>'
            )
            st.markdown(note_html, unsafe_allow_html=True)
        if log_row.get("error_message"):
            error_html = (
                '<div style="background:#1F1315; border:1px solid #4C1D1D; border-radius:8px; '
                'padding:0.75rem 1rem; margin:0.5rem 0; font-size:0.8rem; color:#FCA5A5;">'
                f'<strong>Error at step {log_row["step_index"]}:</strong> {log_row["error_message"]}</div>'
            )
            st.markdown(error_html, unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        sidebar_header_html = (
            '<div style="font-weight:800; font-size:1.05rem; color:#F1F5F9; margin-bottom:0.25rem;">Submit New RFQ</div>'
            '<div style="color:#64748B; font-size:0.78rem; margin-bottom:1.25rem;">Kicks off a new LangGraph thread</div>'
        )
        st.markdown(sidebar_header_html, unsafe_allow_html=True)

        with st.form("rfq_submission_form", clear_on_submit=True):
            client_name = st.text_input("Client Name (optional)")
            client_email = st.text_input("Client Email (optional)")
            rfq_text = st.text_area(
                "RFQ Text",
                height=220,
                placeholder="Paste the inbound RFQ email or document text here…",
            )
            submitted = st.form_submit_button("Submit RFQ", type="primary", use_container_width=True)

            if submitted:
                if not rfq_text or len(rfq_text.strip()) < 10:
                    st.error("RFQ text must be at least 10 characters.")
                else:
                    try:
                        result = submit_rfq_to_backend(rfq_text, client_name, client_email)
                        clear_all_caches()
                        st.success(f"RFQ accepted — task {result['task_id'][:8]}… is running.")
                    except requests.RequestException as exc:
                        st.error(f"Could not reach backend at {_BACKEND_API_BASE_URL}: {exc}")

        st.markdown('<hr class="qf-divider">', unsafe_allow_html=True)
        footer_html = (
            f'<div style="color:#475569; font-size:0.72rem;">Backend: '
            f'<span style="font-family:JetBrains Mono, monospace;">{_BACKEND_API_BASE_URL}</span><br>'
            f'Auto-refresh: every {_REFRESH_SECONDS}s</div>'
        )
        st.markdown(footer_html, unsafe_allow_html=True)

        if st.button("Refresh Now", use_container_width=True):
            clear_all_caches()
            st.rerun()


# ============================================================
# Main layout
# ============================================================

def main() -> None:
    st_autorefresh(interval=_REFRESH_SECONDS * 1000, key="qf_live_refresh")
    render_sidebar()
    render_header()

    summary = load_task_summary()
    render_kpi_grid(summary)

    st.markdown('<div class="qf-section-label">Live Pipeline</div>', unsafe_allow_html=True)
    active_node_counts = load_active_node_counts()
    render_node_pipeline(active_node_counts)

    left_col, right_col = st.columns([3, 2])
    with left_col:
        trend_df = load_confidence_risk_trend()
        render_trend_charts(trend_df)
    with right_col:
        pending_df = load_pending_approvals()
        render_pending_approvals(pending_df)

    st.markdown('<div class="qf-section-label">Operations</div>', unsafe_allow_html=True)
    recent_tasks_df = load_recent_tasks(limit=_MAX_RECENT_LOGS)
    render_recent_tasks_table(recent_tasks_df)
    render_task_inspector(recent_tasks_df)


if __name__ == "__main__":
    main()