"""Ollama Agent module for handling agent operations with locally hosted Ollama models."""

import base64
import os
from typing import Any, Dict, List, Optional, Callable, Union, TypedDict, cast

from .base import BaseAgent, AgentResponse
from .logger import get_logger
from .permissions import PermissionOptions, PermissionRequest, PermissionStatus

# Initialize logger
logger = get_logger(__name__)

# Import Ollama - will be installed as a dependency
try:
    import ollama
    # For type checking, but we avoid direct import to prevent runtime errors
    Message = Any  # Placeholder for ollama.types.Message
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama Python package not found. Please install with 'pip install ollama'")


class ToolCallResult(TypedDict):
    """Type for tool call results"""
    name: str
    parameters: Dict[str, Any]
    output: str
    error: Optional[str]


class OllamaAgent(BaseAgent):
    """
    Ollama Agent that implements the BaseAgent interface using locally hosted Ollama models.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,  # Not used, kept for compatibility
        temperature: float = 0.0,
        timeout: int = 180,
        permission_callback: Optional[Callable[[PermissionRequest], PermissionStatus]] = None,
        permission_options: Optional[PermissionOptions] = None,
        default_tool_timeout: int = 300,
        host: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize an Ollama agent.

        Args:
            model: Ollama model to use (without the "ollama-" prefix)
            api_key: Not used for Ollama, kept for API compatibility
            temperature: Temperature parameter for model (0.0 to 1.0)
            timeout: Timeout in seconds for API requests
            permission_callback: Optional callback for permission requests
            permission_options: Permission configuration options
            default_tool_timeout: Default timeout in seconds for tool calls
            host: Optional Ollama API host URL (default: http://localhost:11434)
            **kwargs: Additional parameters to pass to the model
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama Python package is required. Install with 'pip install ollama'")

        logger.info(f"Initializing Ollama agent with model {model}")

        # Remove "ollama-" prefix if present
        if model.startswith("ollama-"):
            model = model[len("ollama-"):]

        super().__init__(
            api_key=None,
            model=model,
            permission_options=permission_options,
            permission_callback=permission_callback,
            default_tool_timeout=default_tool_timeout
        )

        self.temperature = temperature
        self.timeout = timeout
        self.extra_kwargs = kwargs

        # Get Ollama host with priority: parameter > environment > default
        self.original_host = os.environ.get("OLLAMA_HOST")
        self.host = host or self.original_host or "http://localhost:11434"
        logger.debug(f"Using Ollama host: {self.host}")

        # Set environment variable for Ollama client
        os.environ["OLLAMA_HOST"] = self.host

        # Initialize async client with correct host
        self.async_client = ollama.AsyncClient()
        logger.debug(f"Initialized Ollama client with host: {self.host}")

        self.conversation_history: List[Dict[str, str]] = []
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.system_prompt = self._generate_system_prompt()
        logger.debug(f"Generated system prompt ({len(self.system_prompt)} chars)")
        logger.debug(f"Tool timeouts set to {default_tool_timeout}s")

        # Check Ollama server connection
        try:
            self._check_ollama_server()
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.host}: {str(e)}")
            raise ConnectionError(f"Failed to connect to Ollama server at {self.host}: {str(e)}")

    def __del__(self) -> None:
        """Cleanup when the object is garbage collected."""
        # Restore original OLLAMA_HOST environment variable
        if hasattr(self, 'original_host'):
            if self.original_host is not None:
                os.environ["OLLAMA_HOST"] = self.original_host
            else:
                os.environ.pop("OLLAMA_HOST", None)

    def _check_ollama_server(self) -> None:
        """
        Check if Ollama server is running and accessible.
        Raises ConnectionError if server is not available.
        """
        try:
            # List models using the global client (OLLAMA_HOST env var already set in __init__)
            models = ollama.list()
            available_models: List[str] = []

            # Extract model names from response and handle model tags
            for model_info in models.models:
                model_name = model_info.model
                # Remove tag (e.g., ":latest") if present
                if model_name and ":" in model_name:
                    model_name = model_name.split(":")[0]
                if model_name:
                    available_models.append(model_name)

            if not available_models:
                logger.warning("No models found in Ollama server. Please pull a model first.")
            else:
                logger.debug(f"Available Ollama models: {', '.join(available_models)}")

            # Check if our model is available
            if self.model not in available_models:
                logger.warning(f"Model '{self.model}' not found in available models. "
                               f"You may need to pull it with 'ollama pull {self.model}'")

        except Exception as e:
            logger.error(f"Error checking Ollama server: {str(e)}")
            raise ConnectionError(f"Cannot connect to Ollama server at {self.host}. "
                                  f"Is Ollama running? Error: {str(e)}")

    def _generate_system_prompt(self) -> str:
        """
        Generate the system prompt that defines the agent's capabilities and behavior.
        This prompt is similar to the other agents but adapted for Ollama.
        """
        logger.debug("Generating system prompt for Ollama agent")
        return """
You are a powerful agentic AI coding assistant, powered by a locally hosted Ollama model. You operate exclusively in Cursor, the world's best IDE.

You are pair programming with a USER to solve their coding task.
The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.
Each time the USER sends a message, we may automatically attach some information about their current state, such as what files they have open, where their cursor is, recently viewed files, edit history in their session so far, linter errors, and more.
This information may or may not be relevant to the coding task, it is up for you to decide.
Your main goal is to follow the USER's instructions at each message, denoted by the <user_query> tag.

<tool_calling>
You have tools at your disposal to solve the coding task. Follow these rules regarding tool calls:
1. ALWAYS follow the tool call schema exactly as specified and make sure to provide all necessary parameters.
2. The conversation may reference tools that are no longer available. NEVER call tools that are not explicitly provided.
3. **NEVER refer to tool names when speaking to the USER.** For example, instead of saying 'I need to use the edit_file tool to edit your file', just say 'I will edit your file'.
4. Only calls tools when they are necessary. If the USER's task is general or you already know the answer, just respond without calling tools.
5. Before calling each tool, first explain to the USER why you are calling it.
</tool_calling>

<making_code_changes>
When making code changes, NEVER output code to the USER, unless requested. Instead use one of the code edit tools to implement the change.
Use the code edit tools at most once per turn.
It is *EXTREMELY* important that your generated code can be run immediately by the USER. To ensure this, follow these instructions carefully:
1. Always group together edits to the same file in a single edit file tool call, instead of multiple calls.
2. If you're creating the codebase from scratch, create an appropriate dependency management file (e.g. requirements.txt) with package versions and a helpful README.
3. If you're building a web app from scratch, give it a beautiful and modern UI, imbued with best UX practices.
4. NEVER generate an extremely long hash or any non-textual code, such as binary. These are not helpful to the USER and are very expensive.
5. Unless you are appending some small easy to apply edit to a file, or creating a new file, you MUST read the the contents or section of what you're editing before editing it.
6. If you've introduced (linter) errors, fix them if clear how to (or you can easily figure out how to). Do not make uneducated guesses. And DO NOT loop more than 3 times on fixing linter errors on the same file. On the third time, you should stop and ask the user what to do next.
7. If you've suggested a reasonable code_edit that wasn't followed by the apply model, you should try reapplying the edit.
</making_code_changes>

<searching_and_reading>
You have tools to search the codebase and read files. Follow these rules regarding tool calls:
1. If available, heavily prefer the semantic search tool to grep search, file search, and list dir tools.
2. If you need to read a file, prefer to read larger sections of the file at once over multiple smaller calls.
3. If you have found a reasonable place to edit or answer, do not continue calling tools. Edit or answer from the information you have found.
</searching_and_reading>

Answer the user's request using the relevant tool(s), if they are available. Check that all the required parameters for each tool call are provided or can reasonably be inferred from context. IF there are no relevant tools or there are missing values for required parameters, ask the user to supply these values; otherwise proceed with the tool calls. If the user provides a specific value for a parameter (for example provided in quotes), make sure to use that value EXACTLY. DO NOT make up values for or ask about optional parameters. Carefully analyze descriptive terms in the request as they may indicate required parameter values that should be included even if not explicitly quoted.

You MUST use the following format when citing code regions or blocks:
```12:15:app/components/Todo.tsx
// ... existing code ...
```
This is the ONLY acceptable format for code citations. The format is ```startLine:endLine:filepath where startLine and endLine are line numbers.
"""

    async def chat(self, message: str, user_info: Optional[Dict[str, Any]] = None) -> Union[str, AgentResponse]:
        """
        Send a message to the Ollama model and get a response.

        Args:
            message: The user's message
            user_info: Optional dict containing info about the user's current state

        Returns:
            Either a string response or a structured AgentResponse containing
            the message, tool_calls made, and optional thinking
        """
        formatted_message = self.format_user_message(message, user_info)
        messages = self._prepare_messages(formatted_message)
        # Initialize tools but don't use them yet in this method
        _ = self._prepare_tools()

        try:
            # Call Ollama API with tools
            if self.model:
                response = await self.async_client.chat(
                    model=self.model,
                    messages=cast(Any, messages),
                    options={
                        "temperature": self.temperature,
                        **self.extra_kwargs
                    }
                )

                # Check if response has tool calls (some models might support it)
                has_tool_calls = False
                if hasattr(response, "tool_calls"):
                    has_tool_calls = bool(getattr(response, "tool_calls", None))

                # Add message to conversation history
                self.conversation_history.append({"role": "user", "content": formatted_message})
                content = str(response.message.content) if response.message and hasattr(response.message, "content") else ""
                self.conversation_history.append({"role": "assistant", "content": content})

                if has_tool_calls:
                    # Process and execute tool calls
                    tool_calls_results = self._execute_tool_calls(getattr(response, "tool_calls", []))

                    # Format tool calls for agent response
                    agent_tool_calls = [
                        {
                            "name": result["name"],
                            "parameters": result["parameters"],
                            "output": result["output"],
                            "error": result["error"],
                            "thinking": None
                        }
                        for result in tool_calls_results
                    ]

                    # Return structured agent response
                    return cast(AgentResponse, {
                        "message": content,
                        "tool_calls": agent_tool_calls,
                        "thinking": None
                    })
                else:
                    # Return just the message content for simple responses
                    return content
            else:
                # If model is None, return an error
                return "Error: No model specified for Ollama agent"

        except Exception as e:
            logger.error(f"Error in Ollama chat: {str(e)}")
            return f"Error communicating with Ollama: {str(e)}"

    async def query_image(self, image_paths: List[str], query: str) -> str:
        """
        Query an Ollama model about one or more images.

        Args:
            image_paths: List of paths to local image files
            query: The query/question about the image(s)

        Returns:
            The model's response to the query about the image(s)
        """
        try:
            # Format images for Ollama multimodal input
            content: List[Dict[str, Any]] = [{"type": "text", "text": query}]

            for path in image_paths:
                with open(path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    content.append({"type": "image", "data": image_data})

            # Prepare messages with image content
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]

            # Call Ollama with multimodal content
            if self.model:
                response = await self.async_client.chat(
                    model=self.model,
                    messages=cast(Any, messages),
                    options={
                        "temperature": self.temperature,
                        **self.extra_kwargs
                    }
                )

                # Return just the message content
                return str(response.message.content) if response.message and hasattr(response.message, "content") else ""
            else:
                return "Error: No model specified for Ollama agent"

        except Exception as e:
            error_msg = f"Error in Ollama image query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _prepare_tools(self) -> List[Dict[str, Any]]:
        """
        Format the registered tools for Ollama API.

        Returns:
            Tools in the format expected by Ollama
        """
        if not self.available_tools:
            logger.debug("No tools registered")
            return []

        logger.debug(f"Preparing {len(self.available_tools)} tools for Ollama API")
        tools = []

        for name, tool_data in self.available_tools.items():
            tools.append({
                "name": name,
                "description": tool_data["schema"]["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool_data["schema"]["parameters"]["properties"],
                    "required": tool_data["schema"]["parameters"].get("required", []),
                },
            })
            logger.debug(f"Prepared tool: {name}")

        return tools

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the tool calls made by Ollama.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool call results
        """
        logger.info(f"Executing {len(tool_calls)} tool calls")
        tool_results: List[Dict[str, Any]] = []

        for call in tool_calls:
            try:
                tool_name = call.get("name", "")
                parameters = call.get("parameters", {})

                logger.debug(f"Executing tool: {tool_name} with parameters: {parameters}")

                if tool_name in self.available_tools:
                    # Execute the tool with parameters
                    tool_function = self.available_tools[tool_name]["function"]
                    result = tool_function(**parameters)

                    tool_results.append({
                        "name": tool_name,
                        "parameters": parameters,
                        "output": result.get("output", ""),
                        "error": result.get("error", None)
                    })
                else:
                    error_msg = f"Tool '{tool_name}' not found"
                    logger.warning(error_msg)
                    tool_results.append({
                        "name": tool_name,
                        "parameters": parameters,
                        "output": "",
                        "error": error_msg
                    })
            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}"
                logger.error(error_msg)
                tool_results.append({
                    "name": call.get("name", "unknown"),
                    "parameters": call.get("parameters", {}),
                    "output": "",
                    "error": error_msg
                })

        return tool_results

    def _prepare_messages(self, message: str) -> List[Dict[str, Any]]:
        """
        Prepare message history for Ollama API.

        Args:
            message: The latest user message

        Returns:
            List of messages formatted for Ollama API
        """
        # Start with system message
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)

        # Add current user message
        messages.append({"role": "user", "content": message})

        return messages
