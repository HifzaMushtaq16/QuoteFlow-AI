"""
mcp/tool_schemas.py

QuoteFlow AI v3.0 — MCP Tool Schema Registry

Defines every QuoteFlow AI capability as a standard Model Context Protocol
(MCP) tool schema: a JSON Schema description of the tool's name, purpose,
and input parameters. This registry serves two purposes:

    1. It is the source of truth used by mcp/server.py (or any MCP-compliant
       host) to advertise QuoteFlow AI's capabilities to external agents —
       submitting RFQs, checking task status, resolving human approvals,
       and querying pricing/audit data — without those callers needing any
       knowledge of our internal LangGraph/FastAPI implementation.
    2. It documents, in one place, the exact contract between the outside
       world and backend/main.py's REST endpoints, so schema and API drift
       cannot happen silently.

Every schema strictly follows the MCP tool definition shape:
    { "name": str, "description": str, "inputSchema": <JSON Schema object> }
"""

from __future__ import annotations

from typing import Any, TypedDict


class MCPToolSchema(TypedDict):
    """Canonical shape of a single MCP tool definition."""

    name: str
    description: str
    inputSchema: dict[str, Any]


# ============================================================
# Tool: submit_rfq
# Maps to: POST /api/v1/rfq/submit
# ============================================================

SUBMIT_RFQ_TOOL: MCPToolSchema = {
    "name": "submit_rfq",
    "description": (
        "Submit a new Request for Quote (RFQ) into the QuoteFlow AI autonomous "
        "pipeline. Creates a fresh LangGraph thread that runs the RFQ through "
        "intake parsing, pricing calculation, risk/fraud assessment, and "
        "negotiation — pausing for human approval only if the risk score or "
        "negotiation confidence falls outside configured thresholds. Returns "
        "immediately with a task_id; use get_task_status to poll progress."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "rfq_raw_text": {
                "type": "string",
                "minLength": 10,
                "description": (
                    "The raw inbound RFQ text, exactly as received (e.g. the "
                    "body of an email or document). Must describe the items/"
                    "services being requested and, ideally, quantities."
                ),
            },
            "client_name": {
                "type": ["string", "null"],
                "description": "Client's company or contact name, if already known.",
            },
            "client_email": {
                "type": ["string", "null"],
                "format": "email",
                "description": "Client's email address, if already known.",
            },
        },
        "required": ["rfq_raw_text"],
        "additionalProperties": False,
    },
}


# ============================================================
# Tool: get_task_status
# Maps to: GET /api/v1/rfq/{task_id}/status
# ============================================================

GET_TASK_STATUS_TOOL: MCPToolSchema = {
    "name": "get_task_status",
    "description": (
        "Retrieve the live status of a previously submitted RFQ task directly "
        "from QuoteFlow AI's task_state table — the current LangGraph node, "
        "overall status, latest confidence/risk scores, computed quote amount "
        "and margin, and the finalized PDF URL once available."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The UUID task_id returned by submit_rfq.",
            },
        },
        "required": ["task_id"],
        "additionalProperties": False,
    },
}


# ============================================================
# Tool: get_task_audit_log
# Maps to: GET /api/v1/rfq/{task_id}/audit-log
# ============================================================

GET_TASK_AUDIT_LOG_TOOL: MCPToolSchema = {
    "name": "get_task_audit_log",
    "description": (
        "Fetch the complete, ordered audit trail for an RFQ task — every "
        "LangGraph node execution, the Qwen reasoning summary and confidence/"
        "risk scores at each step, the routing decision made, and latency. "
        "Use this for explainability, debugging, or compliance review of a "
        "specific quote decision."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The UUID task_id whose audit trail should be retrieved.",
            },
        },
        "required": ["task_id"],
        "additionalProperties": False,
    },
}


# ============================================================
# Tool: list_pending_approvals
# Maps to: GET /api/v1/rfq/pending-approval
# ============================================================

LIST_PENDING_APPROVALS_TOOL: MCPToolSchema = {
    "name": "list_pending_approvals",
    "description": (
        "List every RFQ task currently paused at the human_approval_gate node, "
        "awaiting a human reviewer's Approve/Reject decision. A task lands "
        "here when its risk_score meets/exceeds risk_threshold, or its "
        "negotiation confidence_score falls below confidence_threshold."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


# ============================================================
# Tool: resolve_human_approval
# Maps to: POST /api/v1/rfq/{task_id}/approve
# ============================================================

RESOLVE_HUMAN_APPROVAL_TOOL: MCPToolSchema = {
    "name": "resolve_human_approval",
    "description": (
        "Resolve a task paused at human_approval_gate with an Approve or "
        "Reject decision. Approving resumes the LangGraph checkpoint and "
        "proceeds to pdf_quote_generator; rejecting terminates the thread. "
        "This is the only way to advance a task once it reaches "
        "pending_approval status — the graph will not proceed on its own."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The UUID task_id currently in 'pending_approval' status.",
            },
            "decision": {
                "type": "string",
                "enum": ["approved", "rejected"],
                "description": "The human reviewer's decision.",
            },
            "decided_by": {
                "type": "string",
                "minLength": 1,
                "description": "Identifier of the human reviewer making the decision.",
            },
            "reason": {
                "type": ["string", "null"],
                "description": "Optional free-text justification for the decision.",
            },
        },
        "required": ["task_id", "decision", "decided_by"],
        "additionalProperties": False,
    },
}


# ============================================================
# Tool: list_recent_tasks
# Maps to: GET /api/v1/rfq/list
# ============================================================

LIST_RECENT_TASKS_TOOL: MCPToolSchema = {
    "name": "list_recent_tasks",
    "description": (
        "List the most recently updated RFQ tasks across all statuses "
        "(running, pending_approval, approved, rejected, finalized, error), "
        "newest first. Useful for a quick operational overview without "
        "querying a specific task_id."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "default": 50,
                "description": "Maximum number of tasks to return.",
            },
        },
        "additionalProperties": False,
    },
}


# ============================================================
# Registry — full list of tools exposed by the QuoteFlow AI MCP server
# ============================================================

ALL_TOOL_SCHEMAS: list[MCPToolSchema] = [
    SUBMIT_RFQ_TOOL,
    GET_TASK_STATUS_TOOL,
    GET_TASK_AUDIT_LOG_TOOL,
    LIST_PENDING_APPROVALS_TOOL,
    RESOLVE_HUMAN_APPROVAL_TOOL,
    LIST_RECENT_TASKS_TOOL,
]

_TOOL_SCHEMA_BY_NAME: dict[str, MCPToolSchema] = {tool["name"]: tool for tool in ALL_TOOL_SCHEMAS}


def get_tool_schema(tool_name: str) -> MCPToolSchema:
    """
    Look up a single tool's MCP schema by name.

    Args:
        tool_name: The tool's registered name (e.g. "submit_rfq").

    Returns:
        The matching MCPToolSchema.

    Raises:
        KeyError: if no tool with that name is registered.
    """
    if tool_name not in _TOOL_SCHEMA_BY_NAME:
        raise KeyError(
            f"[tool_schemas] Unknown tool '{tool_name}'. "
            f"Registered tools: {list(_TOOL_SCHEMA_BY_NAME.keys())}"
        )
    return _TOOL_SCHEMA_BY_NAME[tool_name]


def list_tool_names() -> list[str]:
    """Return the names of every tool registered in this MCP server."""
    return list(_TOOL_SCHEMA_BY_NAME.keys())


def validate_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> list[str]:
    """
    Perform lightweight, dependency-free validation of arguments against a
    tool's inputSchema (required fields + basic type checks only — this is
    NOT a full JSON Schema validator, just a fast guard used before an MCP
    call is dispatched to the backend).

    Args:
        tool_name: The tool being invoked.
        arguments: The arguments the caller is proposing to pass.

    Returns:
        A list of human-readable validation error strings. Empty list means
        the arguments passed validation.
    """
    schema = get_tool_schema(tool_name)
    input_schema = schema["inputSchema"]
    properties: dict[str, Any] = input_schema.get("properties", {})
    required_fields: list[str] = input_schema.get("required", [])
    errors: list[str] = []

    for field_name in required_fields:
        if field_name not in arguments or arguments[field_name] is None:
            errors.append(f"Missing required field: '{field_name}'")

    if not input_schema.get("additionalProperties", True):
        for provided_field in arguments:
            if provided_field not in properties:
                errors.append(f"Unexpected field not defined in schema: '{provided_field}'")

    for field_name, field_schema in properties.items():
        if field_name not in arguments or arguments[field_name] is None:
            continue
        expected_type = field_schema.get("type")
        if expected_type is None:
            continue
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        expected_types = [t for t in expected_types if t != "null"]
        python_type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict}
        for json_type in expected_types:
            python_type = python_type_map.get(json_type)
            if python_type and not isinstance(arguments[field_name], python_type):
                errors.append(
                    f"Field '{field_name}' expected type '{json_type}', "
                    f"got '{type(arguments[field_name]).__name__}'"
                )

    return errors


if __name__ == "__main__":
    import json as _json

    print(f"QuoteFlow AI v3.0 — {len(ALL_TOOL_SCHEMAS)} MCP tools registered:\n")
    for tool_schema in ALL_TOOL_SCHEMAS:
        print(f"— {tool_schema['name']}")
    print("\nFull schema dump:\n")
    print(_json.dumps(ALL_TOOL_SCHEMAS, indent=2))
