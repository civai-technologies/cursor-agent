"""Consumer contract tests — audience-resolver, pods, nova-api patterns."""

import json
import os
import tempfile
import unittest

from cursor_agent_tools.schemas import (
    build_agent_tool_call,
    build_tool_event_payload,
    enrich_agent_response,
    normalize_tool_result,
    tool_call_result_value,
)
from cursor_agent_tools.tools.file_tools import read_file
from cursor_agent_tools.tools.search_tools import web_search


class TestNormalizeToolResult(unittest.TestCase):
    def test_read_file_content_shape(self) -> None:
        raw = {"content": "hello", "start_line": 1, "end_line": 1, "total_lines": 1}
        norm = normalize_tool_result(raw)
        self.assertTrue(norm["ok"])
        self.assertIn("hello", norm["output"])
        self.assertIsNone(norm["error"])

    def test_edit_file_status_error(self) -> None:
        raw = {"status": "error", "message": "Permission denied"}
        norm = normalize_tool_result(raw)
        self.assertFalse(norm["ok"])
        self.assertEqual(norm["error"], "Permission denied")

    def test_image_tools_result_key(self) -> None:
        raw = {"result": "vision text"}
        norm = normalize_tool_result(raw)
        self.assertTrue(norm["ok"])
        self.assertEqual(norm["output"], "vision text")


class TestDualEmitToolCall(unittest.TestCase):
    def test_build_agent_tool_call_dual_emit(self) -> None:
        norm = normalize_tool_result({"result": '{"queries":[]}'})
        tc = build_agent_tool_call("emit_queries", {"q": "x"}, norm)
        self.assertIn("result", tc)
        self.assertIn("output", tc)
        self.assertEqual(tc["result"], tc["output"])
        result_str = tc["result"]
        assert isinstance(result_str, str)
        json.loads(result_str)

    def test_tool_call_result_value_fallback(self) -> None:
        legacy = {"name": "t", "parameters": {}, "result": "legacy"}
        ollama = {"name": "t", "parameters": {}, "output": "canonical"}
        self.assertEqual(tool_call_result_value(legacy), "legacy")
        self.assertEqual(tool_call_result_value(ollama), "canonical")


class TestAudienceResolverContracts(unittest.TestCase):
    def test_chat_tool_calls_shape(self) -> None:
        norm = normalize_tool_result({"result": '{"queries":["a"]}'})
        tool_calls = [build_agent_tool_call("structured_tool", {}, norm)]
        resp = enrich_agent_response({"message": "ok", "tool_calls": tool_calls, "thinking": None})
        self.assertIsInstance(resp, dict)
        self.assertIn("result", resp["tool_calls"][0])
        data = json.loads(str(resp["tool_calls"][0]["result"]))
        self.assertIn("queries", data)
        self.assertIn("primary_tool_call", resp)

    def test_web_search_results_keys(self) -> None:
        """web_search skip path keeps results array (no API keys required)."""
        ws = web_search(search_term="test", force=False, agent=None)
        self.assertIn("results", ws)
        self.assertIsInstance(ws["results"], list)
        self.assertIn("ok", ws)


class TestPodsHookContract(unittest.TestCase):
    def test_on_tool_event_dual_emit(self) -> None:
        norm = normalize_tool_result({"content": "x"})
        payload = build_tool_event_payload(
            "read_file", {"target_file": "a.txt"}, norm, raw=norm["data"]
        )
        self.assertEqual(payload["tool"], payload["name"])
        self.assertEqual(payload["tool_name"], payload["name"])
        self.assertEqual(payload["args"], payload["parameters"])
        self.assertIn("result", payload)
        self.assertIn("output", payload)
        self.assertIn("ok", payload)


class TestDirectFileToolsContract(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        self.tmp.write("line\n")
        self.tmp.close()

    def tearDown(self) -> None:
        os.unlink(self.tmp.name)

    def test_read_file_direct_return_unchanged(self) -> None:
        result = read_file(self.tmp.name, should_read_entire_file=True)
        self.assertIn("content", result)
        self.assertNotIn("ok", result)


if __name__ == "__main__":
    unittest.main()
