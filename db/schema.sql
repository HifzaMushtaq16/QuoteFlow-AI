-- ============================================================
-- QuoteFlow AI v3.0 — SQLite Schema
-- Orchestration: LangGraph (StateGraph) | Reasoning: Qwen2.5
-- Every LangGraph node transition is persisted here so the
-- Streamlit dashboard can render the live active node in real time.
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- task_state
-- One row per RFQ / LangGraph thread. Mirrors LangGraph's own
-- checkpoint concept (thread_id) so the graph's execution can be
-- resumed, inspected, or replayed independently of the dashboard.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS task_state (
    task_id             TEXT PRIMARY KEY,                  -- UUID for the RFQ
    thread_id           TEXT NOT NULL UNIQUE,               -- LangGraph thread/checkpoint id
    client_name         TEXT,
    client_email        TEXT,
    rfq_raw_text        TEXT,                                -- original inbound email/PDF text
    current_node        TEXT NOT NULL DEFAULT 'intake_parser',
    -- active LangGraph node:
    -- intake_parser | pricing_calculator |
    -- risk_fraud_assessor | negotiation_agent |
    -- human_approval_gate | pdf_quote_generator |
    -- finalized | rejected
    status               TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'pending_approval', 'approved',
                           'rejected', 'finalized', 'error')),
    latest_confidence    REAL,                                -- most recent Qwen confidence_score (0-1)
    latest_risk_score    REAL,                                -- most recent risk_fraud_assessor score (0-1)
    quote_amount         REAL,
    margin_pct           REAL,
    pdf_oss_url          TEXT,                                -- final quote PDF location on Alibaba OSS
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_task_state_status ON task_state(status);
CREATE INDEX IF NOT EXISTS idx_task_state_node ON task_state(current_node);

-- ------------------------------------------------------------
-- audit_log
-- Append-only ledger of every LangGraph node execution.
-- This is the single source of truth the Streamlit dashboard
-- reads to reconstruct "which node is active right now" and to
-- render historical confidence/risk trend charts.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    thread_id           TEXT NOT NULL,
    node_name           TEXT NOT NULL,                       -- LangGraph node that executed
    agent_name          TEXT NOT NULL,                       -- logical agent behind the node
    step_index          INTEGER NOT NULL,                     -- ordinal step within the thread
    input_payload       TEXT,                                  -- JSON blob sent to Qwen / tool
    output_payload      TEXT,                                  -- JSON blob returned by Qwen / tool
    reasoning_summary   TEXT,                                  -- Qwen's free-text reasoning for this step
    confidence_score    REAL,                                  -- Qwen-generated confidence (0-1)
    risk_score          REAL,                                  -- populated only by risk_fraud_assessor
    next_node           TEXT,                                  -- LangGraph conditional-edge decision
    tool_call_name      TEXT,                                  -- MCP-style tool invoked, if any
    latency_ms          INTEGER,
    error_message       TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES task_state(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audit_log_task ON audit_log(task_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_thread ON audit_log(thread_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_node ON audit_log(node_name);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);

-- ------------------------------------------------------------
-- pricing_rules
-- Read by the pricing_calculator node (deterministic tool, not LLM).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pricing_rules (
    rule_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    item_sku            TEXT NOT NULL,
    base_unit_price     REAL NOT NULL,
    min_margin_pct      REAL NOT NULL DEFAULT 0.10,
    max_discount_pct    REAL NOT NULL DEFAULT 0.15,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pricing_rules_sku ON pricing_rules(item_sku);

-- ------------------------------------------------------------
-- clients
-- Read by risk_fraud_assessor node to check history/trust signals.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clients (
    client_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    client_name         TEXT NOT NULL,
    client_email        TEXT UNIQUE,
    trust_score         REAL DEFAULT 0.5,                     -- rolling trust score (0-1)
    total_orders        INTEGER DEFAULT 0,
    flagged_count        INTEGER DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_clients_email ON clients(client_email);

-- ------------------------------------------------------------
-- human_approvals
-- Records each human-in-the-loop decision made from the dashboard.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS human_approvals (
    approval_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    decision            TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    decided_by          TEXT,                                  -- human reviewer identifier
    reason              TEXT,
    decided_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES task_state(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_human_approvals_task ON human_approvals(task_id);

-- ------------------------------------------------------------
-- Seed data: a small default pricing catalog so the
-- pricing_calculator node has deterministic rules to read
-- from on first run (idempotent — safe to re-run schema.sql).
-- ------------------------------------------------------------
INSERT OR IGNORE INTO pricing_rules (rule_id, item_sku, base_unit_price, min_margin_pct, max_discount_pct)
VALUES
    (1, 'SKU-CLOUD-COMPUTE-STD', 120.00, 0.12, 0.15),
    (2, 'SKU-CLOUD-STORAGE-OSS', 0.08,   0.10, 0.10),
    (3, 'SKU-API-CALL-BUNDLE',   250.00, 0.15, 0.20),
    (4, 'SKU-SUPPORT-ENTERPRISE', 899.00, 0.20, 0.10);