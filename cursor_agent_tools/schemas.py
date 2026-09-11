"""Canonical tool/agent boundary schemas — producers own the contract."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TypedDict

# Explicit session/turn exit protocol (single source of truth).
TERMINATE_TOOL_NAME = "terminate_agent_process"
TERMINATE_TEXT_SENTINEL = "terminate_agent_process||"


def is_terminate_tool(tool_name: Any) -> bool:
    """Exact-match check for the terminate tool (no substring matching)."""
    return str(tool_name) == TERMINATE_TOOL_NAME


class ToolResult(TypedDict):
    ok: bool
    output: str
    error: Optional[str]
    data: Optional[Dict[str, Any]]


class AgentToolCallRecord(TypedDict):
    """Dual-emit tool call record returned from agent.chat() tool_calls lists.

    Every producer sets all keys; ``result`` mirrors the legacy LLM-facing
    string (audience-resolver, pods iteration progress read it), while
    ``output``/``error`` are the canonical fields.
    """

    name: str
    parameters: Dict[str, Any]
    result: Optional[str]
    output: str
    error: Optional[str]
    thinking: Optional[str]


class ToolEventPayload(TypedDict, total=False):
    tool: str
    name: str
    args: Dict[str, Any]
    parameters: Dict[str, Any]
    result: Any
    error: Optional[str]
    tool_name: str
    arguments: Dict[str, Any]
    output: str
    ok: bool


def normalize_tool_result(raw: Any) -> ToolResult:
    """Map any tool return dict/str into a canonical ToolResult."""
    if raw is None:
        return {"ok": False, "output": "", "error": "Tool returned None", "data": None}

    if isinstance(raw, str):
        return {"ok": True, "output": raw, "error": None, "data": None}

    if not isinstance(raw, dict):
        return {"ok": True, "output": str(raw), "error": None, "data": None}

    data: Dict[str, Any] = dict(raw)
    err: Optional[str] = None
    ok = True

    if data.get("error"):
        err = str(data["error"])
        ok = False
    elif data.get("status") == "error":
        err = str(data.get("message") or "error")
        ok = False

    output = _serialize_tool_output(data)
    return {"ok": ok, "output": output, "error": err, "data": data}


def _serialize_tool_output(data: Dict[str, Any]) -> str:
    """Produce LLM-facing string content from a tool return dict."""
    if data.get("output") is not None and str(data.get("output")):
        return str(data["output"])
    if data.get("result") is not None:
        result_val = data["result"]
        if isinstance(result_val, str):
            return result_val
        return json.dumps(result_val, default=str)
    return json.dumps(data, default=str)


def llm_tool_content(normalized: ToolResult) -> str:
    """String placed in provider tool-result / tool message channels."""
    if not normalized["ok"] and normalized["error"]:
        data = normalized.get("data")
        if isinstance(data, dict):
            stderr = str(data.get("stderr") or "").strip()
            stdout = str(data.get("stdout") or "").strip()
            if stderr or stdout:
                body = stderr if stderr else stdout
                return f"{normalized['error']}\n{body}"
        return normalized["error"]
    return normalized["output"]


def build_agent_tool_call(
    name: str,
    parameters: Dict[str, Any],
    normalized: ToolResult,
    *,
    thinking: Optional[str] = None,
    llm_content: Optional[str] = None,
) -> AgentToolCallRecord:
    """Dual-emit AgentToolCall: legacy ``result`` + canonical ``output``/``error``."""
    content = llm_content if llm_content is not None else llm_tool_content(normalized)
    return {
        "name": name,
        "parameters": parameters,
        "result": content,
        "output": normalized["output"],
        "error": normalized["error"],
        "thinking": thinking,
    }


def build_tool_event_payload(
    tool_name: str,
    arguments: Dict[str, Any],
    normalized: ToolResult,
    *,
    raw: Any = None,
) -> ToolEventPayload:
    """Dual-emit hook payload for on_tool_event consumers."""
    return {
        "tool": tool_name,
        "name": tool_name,
        "args": arguments,
        "parameters": arguments,
        "result": raw if raw is not None else normalized.get("data"),
        "error": normalized["error"],
        "tool_name": tool_name,
        "arguments": arguments,
        "output": normalized["output"],
        "ok": normalized["ok"],
    }


def tool_call_result_value(tc: Dict[str, Any]) -> Any:
    """Read tool result from agent tool_calls entry (legacy + canonical)."""
    if tc.get("result") is not None:
        return tc.get("result")
    return tc.get("output")


def enrich_agent_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with primary_tool_call (last tool) added — additive, non-mutating."""
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return {**response, "primary_tool_call": tool_calls[-1]}
    return dict(response)
