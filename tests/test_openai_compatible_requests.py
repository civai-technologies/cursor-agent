"""Offline checks for the real OpenAI SDK request serializer."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

from cursor_agent_tools.openai_agent import OpenAIAgent


class TestOpenAICompatibleRequests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_tool_fields_are_omitted_without_registered_tools(self) -> None:
        bodies: List[Dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "id": "chatcmpl-fixture",
                            "created": 1,
                            "object": "chat.completion",
                            "model": "custom-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": "Hi"},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ).encode()
                )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        agent = OpenAIAgent(
            api_key="fixture-key",
            model="custom-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
        try:
            response = await agent.chat("Hello")
            self.assertIsInstance(response, dict)
            assert isinstance(response, dict)
            self.assertEqual(response["message"], "Hi")
            self.assertNotIn("tools", bodies[0])
            self.assertNotIn("tool_choice", bodies[0])
            agent.register_tool(
                "echo",
                lambda text: text,
                "Echo text",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            )
            response = await agent.chat("Hello again")
            assert isinstance(response, dict)
            self.assertEqual(response["message"], "Hi")
            self.assertEqual(bodies[1]["tool_choice"], "auto")
            self.assertEqual(bodies[1]["tools"][0]["function"]["name"], "echo")
        finally:
            await agent.client.close()
            server.shutdown()
            server.server_close()
            thread.join()
