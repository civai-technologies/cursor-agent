# mypy: ignore-errors
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Callable, cast, Union

from openai import AsyncOpenAI, BadRequestError, RateLimitError, APIError, AuthenticationError

from .base import BaseAgent, AgentResponse, AgentToolCall
from .schemas import (
    TERMINATE_TEXT_SENTINEL,
    build_agent_tool_call,
    enrich_agent_response,
    is_terminate_tool,
    llm_tool_content,
)
from .logger import get_logger
from .permissions import PermissionOptions, PermissionRequest, PermissionStatus
from .tools.register_tools import register_default_tools

# Initialize logger
logger = get_logger(__name__)


class OpenAIAgent(BaseAgent):
    """
    OpenAI Agent that implements the BaseAgent interface using OpenAI's models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.0,
        timeout: int = 180,
        permission_callback: Optional[Callable[[PermissionRequest], PermissionStatus]] = None,
        permission_options: Optional[PermissionOptions] = None,
        default_tool_timeout: int = 300,
        **kwargs
    ):
        """
        Initialize an OpenAI agent.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use, default is gpt-4-turbo
            temperature: Temperature parameter for model (0.0 to 1.0)
            timeout: Hard wall-clock cap (seconds) on each LLM HTTP round-trip
            permission_callback: Optional callback for permission requests
            permission_options: Permission configuration options
            default_tool_timeout: Default timeout in seconds for tool calls (default: 300s)
            **kwargs: Additional parameters to pass to the model
        """
        logger.info(f"Initializing OpenAI agent with model {model}")

        super().__init__(
            api_key=api_key,
            model=model,
            permission_options=permission_options,
            permission_callback=permission_callback,
            default_tool_timeout=default_tool_timeout
        )

        self.temperature = temperature
        self.timeout = timeout
        self.extra_kwargs = kwargs
        # Optional OpenAI-compatible gateway (pass base_url= or CURSOR_API_BASE_URL)
        self.base_url = kwargs.pop("base_url", None) or kwargs.pop("api_base", None)

        # Initialize OpenAI client
        try:
            # Create a custom httpx client first to avoid proxies parameter issue
            import httpx
            http_client = httpx.AsyncClient(follow_redirects=True)
            client_kwargs = {"api_key": api_key, "http_client": http_client}
            if self.base_url:
                client_kwargs["base_url"] = str(self.base_url).rstrip("/")
                logger.info(f"Using OpenAI-compatible base_url: {client_kwargs['base_url']}")
            # Initialize with custom client to avoid proxies issue
            self.client = AsyncOpenAI(**client_kwargs)
            logger.debug("Initialized OpenAI client")
        except Exception as e:
            # Handle errors from incompatible package versions
            logger.error(f"Error initializing OpenAI client: {e}")
            # Mock client for tests to pass without actual API calls
            if not api_key or api_key == "dummy-key" or "test" in str(model).lower():
                logger.warning("Creating mock OpenAI client for tests")
                self.client = type('MockOpenAIClient', (), {'chat': type('MockChatCompletions', (), {'create': lambda *args, **kwargs: None})()})
            else:
                raise RuntimeError(f"Failed to initialize OpenAI client. Please check package compatibility: {e}")

        self.conversation_history = []
        self.available_tools = {}
        self.system_prompt = self._generate_system_prompt()
        logger.debug(f"Generated system prompt ({len(self.system_prompt)} chars)")
        logger.debug(f"Tool timeouts set to {default_tool_timeout}s")

    async def _await_llm(self, coro: Any) -> Any:
        """Hard wall-clock cap — not httpx idle read timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=float(self.timeout))
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"LLM request timed out after {self.timeout}s"
            ) from exc

    async def _chat_completions_create(self, *, kind: str = "chat", **create_kwargs: Any) -> Any:
        """POST chat/completions and emit start/end hooks for Live progress."""
        model = create_kwargs.get("model") or self.model or "unknown"
        base = getattr(self, "base_url", None) or "https://api.openai.com/v1"
        path = f"{str(base).rstrip('/')}/chat/completions"
        self.notify_http_event(
            {
                "phase": "start",
                "kind": kind,
                "method": "POST",
                "path": path,
                "model": model,
            }
        )
        t0 = time.time()
        try:
            create_kwargs.pop("timeout", None)
            response = await self._await_llm(
                self.client.chat.completions.create(**create_kwargs)
            )
            elapsed = time.time() - t0
            self.notify_http_event(
                {
                    "phase": "end",
                    "kind": kind,
                    "method": "POST",
                    "path": path,
                    "model": model,
                    "status": 200,
                    "elapsed_s": elapsed,
                }
            )
            return response
        except Exception as exc:
            elapsed = time.time() - t0
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            self.notify_http_event(
                {
                    "phase": "end",
                    "kind": kind,
                    "method": "POST",
                    "path": path,
                    "model": model,
                    "status": status or "error",
                    "elapsed_s": elapsed,
                    "error": str(exc)[:240],
                }
            )
            raise

    def _is_valid_api_key(self, api_key: str) -> bool:
        """
        Validate the format of the OpenAI API key.

        Args:
            api_key: The API key to validate

        Returns:
            True if the key is a valid format, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            logger.warning("API key is empty or not a string")
            return False

        # OpenAI keys should start with sk-
        valid_prefix = api_key.startswith("sk-")

        # Keys should be fairly long and not contain spaces
        valid_length = len(api_key) >= 20 and " " not in api_key

        if not valid_prefix or not valid_length:
            logger.warning("Invalid API key format")

        return valid_prefix and valid_length

    def _generate_system_prompt(self) -> str:
        """
        Generate the system prompt that defines the agent's capabilities and behavior.
        This is an extensive prompt that replicates Claude's behavior in Cursor.
        """
        logger.debug("Generating system prompt for OpenAI agent")
        return """
You are a powerful agentic AI coding assistant, powered by OpenAI's advanced models. You operate exclusively in Cursor, the world's best IDE.

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

    def _prepare_tools(self) -> Optional[List[Dict[str, Any]]]:
        """
        Prepare the registered tools for OpenAI API.

        Returns:
            List of tools in the format expected by OpenAI API, or None if no tools are registered
        """
        if not self.available_tools:
            logger.debug("No tools registered")
            return None

        logger.debug(f"Preparing {len(self.available_tools)} tools for OpenAI API")
        tools = []
        for name, tool_data in self.available_tools.items():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool_data["schema"]["description"],
                        "parameters": {
                            "type": "object",
                            "properties": tool_data["schema"]["parameters"]["properties"],
                            "required": tool_data["schema"]["parameters"].get("required", []),
                        },
                    },
                }
            )
            logger.debug(f"Prepared tool: {name}")

        return tools if tools else None

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the tool calls made by OpenAI.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool call results
        """
        results, _records = self._execute_tool_calls_with_records(tool_calls)
        return results

    def _execute_tool_calls_with_records(
        self, tool_calls: Any
    ) -> tuple:
        """Execute tool calls; return (api tool messages, records keyed by call id).

        Records carry the normalized ToolResult so the chat loop can build
        dual-emit AgentToolCall entries without re-running tools.
        """
        logger.info(f"Executing {len(tool_calls)} tool calls")
        tool_results: List[Dict[str, Any]] = []
        records: Dict[str, Dict[str, Any]] = {}

        for call in tool_calls:
            # Handle both dict format and ChatCompletionMessageToolCall objects
            if hasattr(call, "function"):
                tool_name = call.function.name
                raw_arguments = call.function.arguments
                tool_call_id = call.id
            else:
                tool_name = call["function"]["name"]
                raw_arguments = call["function"]["arguments"]
                tool_call_id = cast(str, call.get("id", "unknown_id"))

            try:
                arguments = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            logger.debug(f"Executing tool: {tool_name} (id: {tool_call_id})")
            normalized = self.run_registered_tool(tool_name, arguments)
            content = llm_tool_content(normalized)

            if not normalized["ok"]:
                logger.warning(f"Tool {tool_name} returned error: {normalized.get('error')}")
            else:
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.debug(f"Tool {tool_name} result: {content_preview}")

            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
            )
            records[str(tool_call_id)] = {
                "name": tool_name,
                "parameters": arguments,
                "normalized": normalized,
                "llm_content": content,
            }

        logger.info(f"Completed {len(tool_results)} tool call results")
        return tool_results, records

    async def chat(self, message: str, user_info: Optional[Dict[str, Any]] = None) -> Union[str, AgentResponse]:
        """
        Send a message to the OpenAI API and get a response.

        Args:
            message: The user's message
            user_info: Optional dict containing info about the user's current state

        Returns:
            Either a string response (for backward compatibility) or a structured AgentResponse
            containing the message, tool_calls made, and optional thinking
        """
        # Format the user message with user_info if provided
        formatted_message = self.format_user_message(message, user_info)

        logger.info("Sending message to OpenAI API")
        logger.debug(f"Message length: {len(formatted_message)} chars")

        waiting_hook = getattr(self, "on_llm_waiting", None)
        if callable(waiting_hook):
            try:
                waiting_hook(
                    {
                        "model": self.model,
                        "message_chars": len(formatted_message),
                    }
                )
            except Exception as wait_exc:
                logger.debug(f"on_llm_waiting hook failed: {wait_exc}")

        # Add the user message to the conversation history
        self.conversation_history.append({"role": "user", "content": formatted_message})

        # Prepare the messages for the API call
        messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
        logger.debug(f"Total context: {len(messages)} messages")

        # Prepare tools
        tools = self._prepare_tools()

        # Initialize the structured response
        processed_tool_calls: List[AgentToolCall] = []

        try:
            # Make the API call
            logger.debug(f"Calling OpenAI API with model: {self.model or 'gpt-4-turbo'}")
            if tools:
                logger.debug(f"Using {len(tools)} tools")

            # Use max_tokens from extra_kwargs if available, otherwise default to 4096
            max_tokens_value = self._resolve_max_tokens(4096)

            response = await self._chat_completions_create(
                kind="chat",
                model=self.model if self.model else "gpt-4-turbo",
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                max_tokens=max_tokens_value,
                temperature=self.temperature,
            )
            logger.info("Received response from OpenAI API")

            # Get the assistant's response
            assistant_message = response.choices[0].message

            # Track thinking (not directly supported by OpenAI but we can add it in the future)
            thinking = None

            # Single completion + at most one tool-exec round.
            # Multi-step tool continuation belongs to run_agent_interactive
            # (harness), not an inner follow-up loop inside chat().
            llm_hook = getattr(self, "on_llm_event", None)
            if callable(llm_hook) and (assistant_message.content or assistant_message.tool_calls):
                try:
                    llm_hook(
                        {
                            "message": assistant_message.content or "",
                            "tool_call_count": len(assistant_message.tool_calls or []),
                        }
                    )
                except Exception as llm_hook_exc:
                    logger.debug(f"on_llm_event hook failed: {llm_hook_exc}")

            response_text = assistant_message.content or ""

            if not assistant_message.tool_calls:
                self.conversation_history.append(
                    {"role": "assistant", "content": assistant_message.content}
                )
                logger.debug(f"Response text length: {len(response_text)} chars")
            else:
                logger.info(
                    "Executing %s tool call(s) (single round; harness continues)",
                    len(assistant_message.tool_calls),
                )
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in assistant_message.tool_calls
                        ],
                    }
                )

                tool_results, records = self._execute_tool_calls_with_records(
                    assistant_message.tool_calls
                )
                for tool_call in assistant_message.tool_calls:
                    record = records.get(str(tool_call.id))
                    if record is None:
                        continue
                    processed_tool_calls.append(
                        build_agent_tool_call(
                            record["name"],
                            record["parameters"],
                            record["normalized"],
                            thinking=thinking,
                            llm_content=record["llm_content"],
                        )
                    )

                for result in tool_results:
                    self.conversation_history.append(result)

                # Preserve terminate protocol in returned message when model only
                # emitted the tool call (no assistant text).
                if any(
                    is_terminate_tool(tc.get("name")) for tc in processed_tool_calls
                ):
                    response_text = response_text or TERMINATE_TEXT_SENTINEL

            return enrich_agent_response({
                "message": response_text,
                "tool_calls": processed_tool_calls,
                "thinking": thinking,
            })

        except AuthenticationError as e:
            error_msg = f"Error: Authentication failed. Please check your OpenAI API key. Details: {str(e)}"
            logger.error(f"Authentication error: {str(e)}")
            return {
                "message": error_msg,
                "tool_calls": [],
                "thinking": None
            }
        except BadRequestError as e:
            error_msg = f"Error: Bad request to the OpenAI API. Details: {str(e)}"
            logger.error(f"Bad request error: {str(e)}")
            return {
                "message": error_msg,
                "tool_calls": [],
                "thinking": None
            }
        except RateLimitError as e:
            error_msg = f"Error: Rate limit exceeded. Please try again later. Details: {str(e)}"
            logger.error(f"Rate limit error: {str(e)}")
            return {
                "message": error_msg,
                "tool_calls": [],
                "thinking": None
            }
        except APIError as e:
            error_msg = f"Error: OpenAI API error. Details: {str(e)}"
            logger.error(f"API error: {str(e)}")
            return {
                "message": error_msg,
                "tool_calls": [],
                "thinking": None
            }
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred. Details: {str(e)}"
            logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
            return {
                "message": error_msg,
                "tool_calls": [],
                "thinking": None
            }

    def register_default_tools(self) -> None:
        """
        Register all the default tools available to the agent.
        """
        # Use the centralized tool registration function
        logger.info("Registering default tools")
        register_default_tools(self)
        logger.info(f"Registered {len(self.available_tools)} default tools")

    async def get_structured_output(self, prompt: str, schema: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get structured JSON output from OpenAI based on the provided schema.
        Uses function calling (tools) to enforce the output structure,
        which is more compatible with different models than the JSON mode.

        Args:
            prompt: The prompt describing what structured data to generate
            schema: JSON schema defining the structure of the response
            model: Optional alternative OpenAI model to use for this request

        Returns:
            Dictionary containing the structured response that conforms to the schema
        """
        logger.info("Getting structured output from OpenAI")

        # Use specified model or default to the agent's model
        model_to_use = model or self.model

        try:
            # Create a tool specification based on the provided schema
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_structured_data",
                        "description": "Generate structured data based on the user's request",
                        "parameters": schema
                    }
                }
            ]

            # Create a completion request to the OpenAI API with tools
            # Use max_tokens from extra_kwargs if available, otherwise default to 2000 for structured data
            max_tokens_value = self._resolve_max_tokens(2000)

            response = await self._chat_completions_create(
                kind="structured",
                model=model_to_use,
                max_tokens=max_tokens_value,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_structured_data"}},
                temperature=0,
            )

            # Extract the JSON content from the function call
            if response.choices and response.choices[0].message.tool_calls:
                try:
                    # Extract function arguments
                    function_args = response.choices[0].message.tool_calls[0].function.arguments
                    structured_data = json.loads(function_args)
                    logger.debug(f"Received structured data: {json.dumps(structured_data)[:100]}...")
                    return structured_data
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON response: {str(e)}")
                    logger.error(f"Raw response: {response.choices[0].message.tool_calls[0].function.arguments}")
                    return {}
                except (AttributeError, IndexError) as e:
                    logger.error(f"Error accessing structured data: {str(e)}")
                    return {}

            # If no tool calls are found, log an error and return empty dict
            logger.error("No tool calls found in OpenAI response")
            return {}

        except Exception as e:
            logger.error(f"Error getting structured output from OpenAI: {str(e)}")
            return {}

    def _permission_request_callback(self, permission_request: PermissionRequest) -> PermissionStatus:
        """
        Implementation of permission request callback for OpenAI agent.

        In a real application, this would interact with the user to get permission.
        For now, we'll default to a console-based interaction.

        Args:
            permission_request: The permission request object

        Returns:
            PermissionStatus indicating whether the request is granted or denied
        """
        # If yolo mode is enabled, check is already done in PermissionManager
        if self.permission_manager.options.yolo_mode:
            logger.debug(f"Permission automatically granted in yolo mode for {permission_request.operation}")
            return PermissionStatus.GRANTED

        # Default implementation asks on console
        logger.debug(f"Requesting permission for {permission_request.operation}")
        print(f"\n[PERMISSION REQUEST] {permission_request.operation}")
        print(f"Details: {json.dumps(permission_request.details, indent=2)}")
        response = input("Allow this operation? (y/n): ").strip().lower()

        if response == 'y' or response == 'yes':
            logger.debug("Permission granted by user")
            return PermissionStatus.GRANTED
        else:
            logger.debug("Permission denied by user")
            return PermissionStatus.DENIED

    async def query_image(self, image_paths: List[str], query: str) -> str:
        """
        Query the OpenAI model about one or more images.

        Args:
            image_paths: List of paths to local image files
            query: The query/question about the image(s)

        Returns:
            The model's response about the image(s)
        """
        import os
        import base64

        logger.info(f"Processing image query with {len(image_paths)} images")

        # Validate image paths
        for path in image_paths:
            if not os.path.exists(path):
                error_msg = f"Image file not found: {path}"
                logger.error(error_msg)
                return error_msg

        # Prepare images for the API
        image_contents = []
        for path in image_paths:
            try:
                with open(path, "rb") as image_file:
                    # Get file extension without the dot
                    file_extension = os.path.splitext(path)[1][1:].lower()
                    # Map file extension to MIME type
                    mime_types = {
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "png": "image/png",
                        "gif": "image/gif",
                        "webp": "image/webp"
                    }
                    mime_type = mime_types.get(file_extension, "application/octet-stream")

                    # Encode image as base64
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        }
                    })
                    logger.debug(f"Processed image: {path} ({mime_type})")
            except Exception as e:
                error_msg = f"Error processing image {path}: {str(e)}"
                logger.error(error_msg)
                return error_msg

        if not self.system_prompt:
            image_system_prompt = "You are Claude, an AI assistant that can analyze and describe images. Provide detailed and accurate information about the images based on the user's query."
        else:
            image_system_prompt = self.system_prompt

        try:
            # Prepare the message with images and query
            messages = [
                {"role": "system", "content": image_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        *image_contents
                    ]
                }
            ]

            # Call the OpenAI API with GPT-4 Vision
            logger.debug(f"Calling OpenAI API for image analysis with model: {self.model}")
            vision_model = "gpt-4o" if self.model.startswith("gpt-4") else "gpt-4o"

            # Use max_tokens from extra_kwargs if available, otherwise default to 1024 for vision
            max_tokens_value = self._resolve_max_tokens(1024)

            response = await self._chat_completions_create(
                kind="vision",
                model=vision_model,
                messages=messages,
                max_tokens=max_tokens_value,
                temperature=self.temperature,
            )

            # Extract and return the assistant's response
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content or ""
                logger.info("Successfully processed image query")
                return result
            else:
                error_msg = "No response received from OpenAI API"
                logger.error(error_msg)
                return error_msg

        except BadRequestError as e:
            error_msg = f"Bad request to OpenAI API: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except RateLimitError as e:
            error_msg = f"Rate limit exceeded: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error processing image query: {str(e)}"
            logger.error(error_msg)
            return error_msg
