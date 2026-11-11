#!/usr/bin/env python3
import asyncio
import os
import shutil
import unittest
from typing import Any, Callable, TypeVar, Optional, ClassVar, Coroutine, Union, List, Dict

import pytest
from unittest.mock import patch

from cursor_agent_tools.ollama_agent import OllamaAgent, OLLAMA_AVAILABLE
from tests.utils import (
    create_user_info,
)
from cursor_agent_tools.base import AgentResponse

# Type variable for the coroutine return type
T = TypeVar('T')


def check_response_quality(response: Union[str, AgentResponse]) -> bool:
    """Check if a response meets basic quality standards."""
    # For structured responses
    if isinstance(response, dict):
        if "message" not in response:
            return False
        message = response["message"]
        # Check if the message is a reasonable length
        return bool(message and len(message) > 20)
    # For string responses (backward compatibility)
    return bool(response and len(response) > 20)


# Helper for async tests
def async_test(coro: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Use asyncio.run() instead of get_event_loop()
        return asyncio.run(coro(*args, **kwargs))
    return wrapper


@pytest.mark.ollama
class TestOllamaAgent(unittest.TestCase):
    """Test the Ollama agent functionality with real local Ollama server."""

    # Test against a small model by default - users can change this in their env
    test_model: ClassVar[str] = os.environ.get("OLLAMA_TEST_MODEL", "llama3.2")
    host: ClassVar[str] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def setUp(self) -> None:
        """Set up the test environment."""
        # Skip if ollama package is not available
        if not OLLAMA_AVAILABLE:
            self.skipTest("Ollama Python package not installed")

        print("\n=== TEST SETUP ===")
        print(f"Using Ollama test model: {self.test_model}")
        print(f"Using Ollama host: {self.host}")

        # Store the original directory to ensure we can return to it
        try:
            self.original_dir = os.path.abspath(os.getcwd())
        except FileNotFoundError:
            # If we can't get the current directory, use the directory where the test file is located
            self.original_dir = os.path.abspath(os.path.dirname(__file__))
            os.chdir(self.original_dir)
            print(f"Reset working directory to: {self.original_dir}")

        self.env: Dict[str, Any] = {}  # Initialize as empty dict
        print("Test environment setup complete")

        # Initialize Ollama agent
        print(f"Initializing Ollama agent with model: ollama-{self.test_model}")
        try:
            self.agent = OllamaAgent(
                model=f"ollama-{self.test_model}",
                host=self.host
            )
            print("Agent initialized successfully")
            print("Registering default tools")
            self.agent.register_default_tools()
            print(f"Default tools registered: {len(self.agent.available_tools)} tools available")
        except Exception as e:
            print(f"Failed to initialize Ollama agent: {str(e)}")
            self.skipTest(f"Failed to initialize Ollama agent: {str(e)}")

        # Debug information - use try/except to handle potential errors
        try:
            print(f"Current working directory: {os.getcwd()}")

            # Create an absolute path for the test directory
            self.test_dir = os.path.abspath(os.path.join(self.original_dir, "test_files_tmp"))

            # Ensure directory exists
            os.makedirs(self.test_dir, exist_ok=True)

            print(f"Test directory path: {self.test_dir}")
            print(f"Test directory exists: {os.path.exists(self.test_dir)}")
            print(f"Test directory is writable: {os.access(self.test_dir, os.W_OK)}")

            # Test that we can actually write to the directory
            test_file = os.path.join(self.test_dir, "test_permissions.txt")
            with open(test_file, 'w') as f:
                f.write("Test file to verify permissions\n")
            os.remove(test_file)  # Clean up
            print("Test directory write permissions verified")

        except Exception as e:
            print(f"Error setting up test environment: {str(e)}")
            self.skipTest(f"Error setting up test environment: {str(e)}")

        print("=== SETUP COMPLETE ===\n")

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Clean up test directory if it exists
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not clean up test directory: {str(e)}")

        # Return to original directory if we changed it
        if hasattr(self, "original_dir"):
            try:
                os.chdir(self.original_dir)
            except Exception as e:
                print(f"Warning: Could not return to original directory: {str(e)}")

    def test_init(self) -> None:
        """Test agent initialization."""
        print("\n--- Testing agent initialization ---")

        print(f"Creating new agent instance with model: ollama-{self.test_model}")
        agent = OllamaAgent(model=f"ollama-{self.test_model}", host=self.host)

        print(f"Agent type: {type(agent)}")
        print(f"Model name: {agent.model} (original: ollama-{self.test_model})")
        print(f"Host: {agent.host}")
        print(f"System prompt length: {len(agent.system_prompt)} characters")

        self.assertIsNotNone(agent)
        self.assertEqual(agent.model, self.test_model)  # Check prefix was removed

        print("--- Agent initialization test passed ---\n")

    def test_tool_registration(self) -> None:
        """Test registering tools."""
        print("\n--- Testing tool registration ---")

        print(f"Creating new agent instance with model: ollama-{self.test_model}")
        agent = OllamaAgent(model=f"ollama-{self.test_model}", host=self.host)

        test_tool_name = "test_tool"
        test_tool_description = "Test tool"

        print(f"Registering custom tool: {test_tool_name}")
        print(f"Tool description: {test_tool_description}")

        # Simple mock function that returns its input
        def test_function(input: str) -> dict:
            return {"output": input, "error": None}

        agent.register_tool(
            name=test_tool_name,
            function=test_function,
            description=test_tool_description,
            parameters={
                "properties": {
                    "input": {"description": "Test input", "type": "string"}
                },
                "required": ["input"]
            }
        )

        available_tools = agent.available_tools
        print(f"Agent has {len(available_tools)} tools registered")
        print(f"Registered tool names: {', '.join(available_tools.keys())}")

        if test_tool_name in available_tools:
            tool_data = available_tools[test_tool_name]
            print(f"Tool schema: {tool_data['schema']}")
            print(f"Tool description in schema: {tool_data['schema']['description']}")

        self.assertIn(test_tool_name, agent.available_tools)
        self.assertEqual(agent.available_tools[test_tool_name]["schema"]["description"], test_tool_description)

        print("--- Tool registration test passed ---\n")

    @async_test
    async def test_simple_query(self) -> None:
        """Test a simple query without tools."""
        # Skip if Ollama server isn't running or model isn't available
        try:
            # Ensure OLLAMA_HOST is set
            os.environ["OLLAMA_HOST"] = self.host

            # Check model availability
            import ollama
            ollama_response = ollama.list()

            # Extract model names and normalize them (removing tags)
            available_models: List[str] = []
            for model_info in ollama_response.models:
                model_name = model_info.model
                if model_name and ":" in model_name:
                    model_name = model_name.split(":")[0]
                if model_name:
                    available_models.append(model_name)

            # Check if test model is available
            if self.test_model not in available_models:
                self.skipTest(f"Model {self.test_model} not available in Ollama server")

        except Exception as e:
            self.skipTest(f"Could not connect to Ollama server: {str(e)}")

        query = "What is the capital of France?"
        print(f"\n--- Testing simple query: '{query}' ---")
        response: Union[str, AgentResponse] = await self.agent.chat(query)

        # Print response for debugging
        if isinstance(response, dict):
            print(f"Response type: Dict with keys {list(response.keys())}")
            if response.get("message"):
                message_content = str(response.get("message", ""))
                print(f"Message content (truncated): {message_content[:100]}...")
            if response.get("tool_calls"):
                print(f"Tool calls: {len(response['tool_calls'])} tool(s) called")
                for i, tool_call in enumerate(response["tool_calls"]):
                    print(f"  Tool {i + 1}: {tool_call.get('name')} with params: {tool_call.get('parameters')}")
        else:
            resp_str = str(response)
            print("Response type: String")
            print(f"Response content (truncated): {resp_str[:100]}...")

        # Check if it's the new structured response
        if isinstance(response, dict):
            self.assertIn("message", response)
            self.assertIsInstance(response["message"], str)
            self.assertIn("tool_calls", response)

            # Check if there's actual content in the message
            self.assertTrue(check_response_quality(response))
        else:
            # For backward compatibility with string responses
            self.assertIsInstance(response, str)
            self.assertTrue(check_response_quality(response))

        print("--- Simple query test passed ---\n")

    @async_test
    async def test_chat_with_user_info(self) -> None:
        """Test chat with user info."""
        # Skip if Ollama server isn't running or model isn't available
        try:
            # Ensure OLLAMA_HOST is set
            os.environ["OLLAMA_HOST"] = self.host

            # Check model availability
            import ollama
            ollama_response = ollama.list()

            # Extract model names and normalize them (removing tags)
            available_models: List[str] = []
            for model_info in ollama_response.models:
                model_name = model_info.model
                if model_name and ":" in model_name:
                    model_name = model_name.split(":")[0]
                if model_name:
                    available_models.append(model_name)

            # Check if test model is available
            if self.test_model not in available_models:
                self.skipTest(f"Model {self.test_model} not available in Ollama server")

        except Exception as e:
            self.skipTest(f"Could not connect to Ollama server: {str(e)}")

        query = "What files do I have open?"
        user_info = create_user_info()

        print(f"\n--- Testing chat with user info: '{query}' ---")
        print(f"User info provided: {type(user_info)} with keys {list(user_info.keys()) if user_info else 'None'}")

        chat_response: Union[str, AgentResponse] = await self.agent.chat(query, user_info)

        # Print response for debugging
        if isinstance(chat_response, dict):
            print(f"Response type: Dict with keys {list(chat_response.keys())}")
            if chat_response.get("message"):
                message_content = str(chat_response.get("message", ""))
                print(f"Message content (truncated): {message_content[:100]}...")
            if chat_response.get("tool_calls"):
                print(f"Tool calls: {len(chat_response['tool_calls'])} tool(s) called")
                for i, tool_call in enumerate(chat_response["tool_calls"]):
                    print(f"  Tool {i + 1}: {tool_call.get('name')} with params: {tool_call.get('parameters')}")
        else:
            resp_str = str(chat_response)
            print("Response type: String")
            print(f"Response content (truncated): {resp_str[:100]}...")

        self.assertTrue(check_response_quality(chat_response))

        # We don't necessarily know the exact content that will be returned,
        # so we just check that there's a reasonable response
        if isinstance(chat_response, dict):
            self.assertIn("message", chat_response)
        else:
            self.assertIsInstance(chat_response, str)

        print("--- Chat with user info test passed ---\n")

    @async_test
    async def test_connection_error(self) -> None:
        """Test behavior when Ollama server is not available."""
        print("\n--- Testing connection error handling ---")

        with patch('ollama.AsyncClient.chat') as mock_chat:
            mock_chat.side_effect = ConnectionError("Could not connect to Ollama server")

            agent = OllamaAgent(model=f"ollama-{self.test_model}", host="http://nonexistent:11434")
            response = await agent.chat("Hello, how are you?")

            print(f"Response when server unavailable: {response}")

            self.assertIsInstance(response, str)
            self.assertIn("Error communicating with Ollama", response)

        print("--- Connection error test passed ---\n")

    @unittest.skip("Only run if you need to test vision capabilities")
    @async_test
    async def test_image_query(self) -> None:
        """Test image query capabilities."""
        print("\n--- Testing image query capabilities ---")

        # Skip if Ollama server isn't running or model isn't available/has vision capabilities
        try:
            # Ensure OLLAMA_HOST is set
            os.environ["OLLAMA_HOST"] = self.host

            # Check model availability
            import ollama
            ollama_response = ollama.list()

            # Extract model names and normalize them (removing tags)
            available_models: List[str] = []
            for model_info in ollama_response.models:
                model_name = model_info.model
                if model_name and ":" in model_name:
                    model_name = model_name.split(":")[0]
                if model_name:
                    available_models.append(model_name)

            # Check if test model is available
            if self.test_model not in available_models:
                self.skipTest(f"Model {self.test_model} not available in Ollama server")

            print(f"Using model: {self.test_model}")
        except Exception as e:
            self.skipTest(f"Could not connect to Ollama server: {str(e)}")

        # Create a test image file
        image_path = os.path.join(self.test_dir, "test_image.jpg")
        # This would need a real test image

        if not os.path.exists(image_path):
            self.skipTest("Test image not available")

        print(f"Using image at: {image_path}")

        response_str = await self.agent.query_image([image_path], "What's in this image?")

        print(f"Image query response (truncated): {response_str[:100]}...")

        self.assertTrue(check_response_quality(response_str))

        print("--- Image query test passed ---\n")
