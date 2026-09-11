"""Loop-level tool execution tests — real agents, real tool functions.

These exercise the provider tool-execution paths (_execute_tool_calls_with_records)
that chat() uses, so a regression like "missing tool record" can never pass CI
silently again. No mocks: registered tools are real Python functions, and the
live chat tests run against real APIs when keys are present.
"""

import asyncio
import json
import os
import unittest
from typing import Any, Dict

from cursor_agent_tools.claude_agent import ClaudeAgent
from cursor_agent_tools.openai_agent import OpenAIAgent
from cursor_agent_tools.schemas import (
    TERMINATE_TEXT_SENTINEL,
    TERMINATE_TOOL_NAME,
    is_terminate_tool,
)

ADD_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "integer"},
        "b": {"type": "integer"},
    },
    "required": ["a", "b"],
}


def add_numbers(a: int, b: int) -> Dict[str, Any]:
    """Real tool function — actually computes."""
    return {"output": str(a + b), "error": None}


def failing_tool() -> Dict[str, Any]:
    """Real tool function that raises a real exception."""
    raise RuntimeError("real failure")


class TestOpenAIToolExecutionRecords(unittest.TestCase):
    def setUp(self) -> None:
        key = os.environ.get("OPENAI_API_KEY") or "sk-dummy-key-for-offline-tool-exec"
        self.agent = OpenAIAgent(api_key=key, model="gpt-4o")
        self.agent.register_tool("add_numbers", add_numbers, "Add two numbers", ADD_SCHEMA)
        self.agent.register_tool(
            "failing_tool", failing_tool, "Always fails",
            {"type": "object", "properties": {}, "required": []},
        )

    def test_records_populated_and_keyed_by_call_id(self) -> None:
        calls = [
            {
                "id": "call_1",
                "function": {"name": "add_numbers", "arguments": json.dumps({"a": 2, "b": 3})},
            }
        ]
        results, records = self.agent._execute_tool_calls_with_records(calls)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tool_call_id"], "call_1")
        self.assertEqual(results[0]["content"], "5")
        record = records["call_1"]
        self.assertEqual(record["name"], "add_numbers")
        self.assertTrue(record["normalized"]["ok"])
        self.assertEqual(record["llm_content"], "5")

    def test_tool_exception_normalized_not_raised(self) -> None:
        calls = [{"id": "call_err", "function": {"name": "failing_tool", "arguments": "{}"}}]
        results, records = self.agent._execute_tool_calls_with_records(calls)
        self.assertEqual(len(results), 1)
        record = records["call_err"]
        self.assertFalse(record["normalized"]["ok"])
        self.assertIn("real failure", str(record["normalized"]["error"]))

    def test_unknown_tool_produces_error_record(self) -> None:
        calls = [{"id": "call_x", "function": {"name": "no_such_tool", "arguments": "{}"}}]
        results, records = self.agent._execute_tool_calls_with_records(calls)
        self.assertEqual(len(results), 1)
        self.assertFalse(records["call_x"]["normalized"]["ok"])

    def test_terminate_tool_real_output(self) -> None:
        self.agent.register_default_tools()  # terminate tool ships with defaults
        calls = [
            {"id": "call_t", "function": {"name": TERMINATE_TOOL_NAME, "arguments": "{}"}}
        ]
        results, records = self.agent._execute_tool_calls_with_records(calls)
        self.assertIn(TERMINATE_TEXT_SENTINEL, results[0]["content"])
        self.assertTrue(is_terminate_tool(records["call_t"]["name"]))


class TestClaudeToolExecutionRecords(unittest.TestCase):
    def setUp(self) -> None:
        key = os.environ.get("ANTHROPIC_API_KEY") or "sk-ant-dummy"
        self.agent = ClaudeAgent(api_key=key)
        self.agent.register_tool("add_numbers", add_numbers, "Add two numbers", ADD_SCHEMA)

    def test_records_and_tool_result_blocks(self) -> None:
        calls = [{"name": "add_numbers", "id": "toolu_1", "input": {"a": 10, "b": 4}}]
        results, records = self.agent._execute_tool_calls_with_records(calls)
        self.assertEqual(len(results), 1)
        block = results[0]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "toolu_1")
        self.assertEqual(block["content"], "14")
        self.assertNotIn("is_error", block)
        record = records["toolu_1"]
        self.assertTrue(record["normalized"]["ok"])
        self.assertEqual(record["llm_content"], "14")

    def test_error_marks_is_error_block(self) -> None:
        calls = [{"name": "missing_tool", "id": "toolu_2", "input": {}}]
        results, records = self.agent._execute_tool_calls_with_records(calls)
        block = results[0]["content"][0]
        self.assertTrue(block.get("is_error"))
        self.assertFalse(records["toolu_2"]["normalized"]["ok"])


class TestTerminateSemantics(unittest.TestCase):
    def test_equality_not_substring(self) -> None:
        self.assertTrue(is_terminate_tool(TERMINATE_TOOL_NAME))
        self.assertFalse(is_terminate_tool("my_terminate_agent_process_wrapper"))
        self.assertFalse(is_terminate_tool("terminate"))


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestOpenAILiveChatToolLoop(unittest.TestCase):
    """Live end-to-end: real API call through chat() with a real tool."""

    def test_chat_executes_tool_and_returns_records(self) -> None:
        agent = OpenAIAgent(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
        agent.register_tool("add_numbers", add_numbers, "Add two integers", ADD_SCHEMA)
        response = asyncio.run(
            agent.chat("Use the add_numbers tool to add 17 and 25, then report the sum.")
        )
        self.assertIsInstance(response, dict)
        assert isinstance(response, dict)
        message = str(response.get("message") or "")
        if message.startswith("Error:") and not response.get("tool_calls"):
            self.skipTest(f"OpenAI live call unavailable: {message[:160]}")
        tool_calls = response["tool_calls"]
        self.assertTrue(tool_calls, "model should have called add_numbers")
        add_calls = [tc for tc in tool_calls if tc["name"] == "add_numbers"]
        self.assertTrue(add_calls)
        # Dual-emit contract: legacy result and canonical output both set
        self.assertEqual(add_calls[0]["result"], add_calls[0]["output"])
        self.assertIn("42", add_calls[0]["output"])


@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set")
class TestClaudeLiveChatToolLoop(unittest.TestCase):
    """Live end-to-end: real API call through chat() with a real tool."""

    def test_chat_executes_tool_and_returns_records(self) -> None:
        agent = ClaudeAgent(api_key=os.environ["ANTHROPIC_API_KEY"])
        agent.register_tool("add_numbers", add_numbers, "Add two integers", ADD_SCHEMA)
        response = asyncio.run(
            agent.chat("Use the add_numbers tool to add 17 and 25, then report the sum.")
        )
        self.assertIsInstance(response, dict)
        assert isinstance(response, dict)
        message = str(response.get("message") or "")
        if message.startswith("Error:") and not response.get("tool_calls"):
            # Billing / auth / quota — not a package regression
            self.skipTest(f"Anthropic live call unavailable: {message[:160]}")
        tool_calls = response["tool_calls"]
        self.assertTrue(tool_calls, "model should have called add_numbers")
        add_calls = [tc for tc in tool_calls if tc["name"] == "add_numbers"]
        self.assertTrue(add_calls)
        self.assertEqual(add_calls[0]["result"], add_calls[0]["output"])
        self.assertIn("42", add_calls[0]["output"])


if __name__ == "__main__":
    unittest.main()
