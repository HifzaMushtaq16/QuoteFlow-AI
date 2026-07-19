"""
mcp/server.py

QuoteFlow AI v3.0 — MCP Server

Exposes QuoteFlow AI's autonomous quoting pipeline as a standard Model
Context Protocol (MCP) server over stdio, so any MCP-compatible client
(Claude Desktop, other agent frameworks, etc.) can submit RFQs, check task
status, review audit trails, and resolve human approvals — without any
knowledge of our internal LangGraph/FastAPI implementation.

This server is a thin dispatch layer only: every tool call is validated
against mcp/tool_schemas.py, then forwarded as an HTTP request to the
running FastAPI backend (backend/main.py). No business logic, database
access, or LangGraph invocation happens in this file directly — the
FastAPI backend remains the single entrypoint for all pipeline mutations
and reads, per QuoteFlow AI's architecture.

Run with:
    python mcp/server.py

Configure the target backend via the QUOTEFLOW_API_BASE_URL environment
variable (defaults to http://localhost:8000).
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Ensure the project root is importable when this file is run directly
# (e.g. `python mcp/server.py` from the project root).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp_app.tool_schemas import (  # noqa: E402
    ALL_TOOL_SCHEMAS,
    get_tool_schema,
    validate_tool_arguments,
)

load_dotenv()

_BACKEND_API_BASE_URL = os.environ.get("QUOTEFLOW_API_BASE_URL", "http://localhost:8000")
_REQUEST_TIMEOUT_SECONDS = 20


class MCPDispatchError(Exception):
    """Raised when a tool call cannot be validated or dispatched to the backend."""


# ============================================================
# Tool dispatch — maps each MCP tool name to a backend HTTP call
# ============================================================

def _dispatch_submit_rfq(arguments: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "rfq_raw_text": arguments["rfq_raw_text"],
        "client_name": arguments.get("client_name"),
        "client_email": arguments.get("client_email"),
    }
    response = requests.post(
        f"{_BACKEND_API_BASE_URL}/api/v1/rfq/submit",
        json=payload,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _dispatch_get_task_status(arguments: dict[str, Any]) -> dict[str, Any]:
    task_id = arguments["task_id"]
    response = requests.get(
        f"{_BACKEND_API_BASE_URL}/api/v1/rfq/{task_id}/status",
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _dispatch_get_task_audit_log(arguments: dict[str, Any]) -> dict[str, Any]:
    task_id = arguments["task_id"]
    response = requests.get(
        f"{_BACKEND_API_BASE_URL}/api/v1/rfq/{task_id}/audit-log",
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return {"audit_log": response.json()}


def _dispatch_list_pending_approvals(arguments: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(
        f"{_BACKEND_API_BASE_URL}/api/v1/rfq/pending-approval",
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return {"pending_approvals": response.json()}


def _dispatch_resolve_human_approval(arguments: dict[str, Any]) -> dict[str, Any]:
    task_id = arguments["task_id"]
    payload = {
        "decision": arguments["decision"],
        "decided_by": arguments["decided_by"],
        "reason": arguments.get("reason"),
    }
    response = requests.post(
        f"{_BACKEND_API_BASE_URL}/api/v1/rfq/{task_id}/approve",
        json=payload,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _dispatch_list_recent_tasks(arguments: dict[str, Any]) -> dict[str, Any]:
    limit = arguments.get("limit", 50)
    response = requests.get(
        f"{_BACKEND_API_BASE_URL}/api/v1/rfq/list",
        params={"limit": limit},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return {"tasks": response.json()}


_TOOL_DISPATCH_TABLE = {
    "submit_rfq": _dispatch_submit_rfq,
    "get_task_status": _dispatch_get_task_status,
    "get_task_audit_log": _dispatch_get_task_audit_log,
    "list_pending_approvals": _dispatch_list_pending_approvals,
    "resolve_human_approval": _dispatch_resolve_human_approval,
    "list_recent_tasks": _dispatch_list_recent_tasks,
}


def dispatch_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Validate arguments against the tool's registered schema, then dispatch
    to the corresponding FastAPI backend endpoint.

    Args:
        tool_name: The MCP tool name being invoked.
        arguments: The arguments supplied by the calling MCP client.

    Returns:
        The parsed JSON response from the backend, ready to be serialized
        back to the MCP client as tool output.

    Raises:
        MCPDispatchError: if the tool is unknown, arguments fail validation,
            or the backend call fails.
    """
    if tool_name not in _TOOL_DISPATCH_TABLE:
        raise MCPDispatchError(f"[mcp_server] Unknown tool: '{tool_name}'")

    validation_errors = validate_tool_arguments(tool_name, arguments)
    if validation_errors:
        raise MCPDispatchError(
            f"[mcp_server] Invalid arguments for '{tool_name}': {'; '.join(validation_errors)}"
        )

    try:
        return _TOOL_DISPATCH_TABLE[tool_name](arguments)
    except requests.exceptions.ConnectionError as exc:
        raise MCPDispatchError(
            f"[mcp_server] Could not reach QuoteFlow AI backend at {_BACKEND_API_BASE_URL}. "
            f"Is `uvicorn backend.main:app` running? Details: {exc}"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        detail = exc.response.json().get("detail") if exc.response is not None else str(exc)
        raise MCPDispatchError(
            f"[mcp_server] Backend returned HTTP {status_code} for '{tool_name}': {detail}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise MCPDispatchError(f"[mcp_server] Request to backend failed for '{tool_name}': {exc}") from exc


# ============================================================
# MCP Server wiring
# ============================================================

app = Server("quoteflow-ai")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise every QuoteFlow AI capability as an MCP Tool definition."""
    return [
        Tool(
            name=schema["name"],
            description=schema["description"],
            inputSchema=schema["inputSchema"],
        )
        for schema in ALL_TOOL_SCHEMAS
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    """
    Handle an incoming MCP tool call: validate, dispatch to the FastAPI
    backend, and return the result (or a clear error) as text content.
    """
    import json

    arguments = arguments or {}

    try:
        # Ensure the requested tool actually exists before dispatching, so
        # unknown tool names surface a clean MCP-level error rather than a
        # raw KeyError from the dispatch table.
        get_tool_schema(name)
        result = dispatch_tool_call(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except (MCPDispatchError, KeyError) as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


async def run_server() -> None:
    """Start the MCP server, communicating over stdio per the MCP spec."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="quoteflow-ai",
                server_version="3.0.0",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    print(
        f"[mcp_server] Starting QuoteFlow AI MCP server "
        f"(backend target: {_BACKEND_API_BASE_URL})...",
        file=sys.stderr,
    )
    asyncio.run(run_server())