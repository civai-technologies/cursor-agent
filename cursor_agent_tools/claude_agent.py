# mypy: ignore-errors
import json
from typing import Any, Dict, List, Optional, Callable, Union

from anthropic import APIError, AsyncAnthropic, AuthenticationError, BadRequestError, RateLimitError

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


class ClaudeAgent(BaseAgent):
    """
    Claude Agent that implements the BaseAgent interface using Anthropic's Claude models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-latest",
        temperature: float = 0.0,
        timeout: int = 180,
        permission_callback: Optional[Callable[[PermissionRequest], PermissionStatus]] = None,
        permission_options: Optional[PermissionOptions] = None,
        default_tool_timeout: int = 300,
        **kwargs
    ):
        """
        Initialize a Claude agent.

        Args:
            api_key: Anthropic API key
            model: Claude model to use, default is claude-3-opus
            temperature: Temperature parameter for model (0.0 to 1.0)
            timeout: Timeout in seconds for API requests
            permission_callback: Optional callback for permission requests
            permission_options: Permission configuration options
            default_tool_timeout: Default timeout in seconds for tool calls (default: 300s)
            **kwargs: Additional parameters to pass to the model
        """
        logger.info(f"Initializing Claude agent with model {model}")

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

        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=api_key)
        logger.debug("Initialized Anthropic client")

        self.conversation_history = []
        self.available_tools = {}
        self.system_prompt = self._generate_system_prompt()
        logger.debug(f"Generated system prompt ({len(self.system_prompt)} chars)")
        logger.debug(f"Tool timeouts set to {default_tool_timeout}s")

    def _is_valid_api_key(self, api_key: str) -> bool:
        """
        Validate the format of the Anthropic API key.

        Args:
            api_key: The API key to validate

        Returns:
            True if the key is a valid format, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            logger.warning("API key is empty or not a string")
            return False

        # Allow dummy keys in test environments
        if api_key == "sk-ant-dummy" or "dummy" in api_key:
            logger.debug("Using dummy API key for testing")
            return True

        # Anthropic keys should start with sk-ant- or sk-
        valid_prefix = api_key.startswith(("sk-ant-", "sk-"))

        # Keys should be fairly long and not contain spaces
        valid_length = len(api_key) >= 20 and " " not in api_key

        if not valid_prefix or not valid_length:
            logger.warning("Invalid API key format")

        return valid_prefix and valid_length

    def _generate_system_prompt(self) -> str:
        """
        Generate the system prompt that defines Claude's capabilities and behavior.
        This is an extensive prompt that replicates Claude's behavior in Cursor.
        """
        logger.debug("Generating system prompt for Claude agent")
        return """
You are a powerful agentic AI coding assistant, powered by Claude 3.7 Sonnet. You operate exclusively in Cursor, the world's best IDE.

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
        Prepare the registered tools for Claude API.

        Returns:
            List of tools in the format expected by Claude API, or None if no tools are registered
        """
        if not self.available_tools:
            logger.debug("No tools registered")
            return None

        logger.debug(f"Preparing {len(self.available_tools)} tools for Claude API")
        tools = []
        for name, tool_data in self.available_tools.items():
            # Claude tools format:
            # {
            #   "name": "tool_name",
            #   "description": "Tool description",
            #   "input_schema": {
            #     "type": "object",
            #     "properties": {
            #       "property_name": {
            #         "type": "string",
            #         "description": "Property description"
            #       },
            #       ...
            #     },
            #     "required": ["property_name", ...]
            #   }
            # }

            tool = {
                "name": name,
                "description": tool_data["schema"]["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool_data["schema"]["parameters"]["properties"],
                    "required": tool_data["schema"]["parameters"].get("required", []),
                },
            }
            tools.append(tool)
            logger.debug(f"Prepared tool: {name}")

        return tools if tools else None

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the tool calls made by Claude.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool call results formatted for the Claude API
        """
        results, _records = self._execute_tool_calls_with_records(tool_calls)
        return results

    def _execute_tool_calls_with_records(
        self, tool_calls: List[Dict[str, Any]]
    ) -> tuple:
        """Execute tool calls; return (Claude tool_result messages, records by id).

        Records carry the normalized ToolResult so the chat flow can build
        dual-emit AgentToolCall entries without re-running tools.
        """
        logger.info(f"Executing {len(tool_calls)} tool calls")
        tool_results: List[Dict[str, Any]] = []
        records: Dict[str, Dict[str, Any]] = {}

        for call in tool_calls:
            tool_name = call["name"]
            tool_id = call.get("id")
            arguments = call.get("input", {})
            if not isinstance(arguments, dict):
                arguments = {}

            logger.debug(f"Executing tool: {tool_name} (id: {tool_id})")
            normalized = self.run_registered_tool(tool_name, arguments)
            content = llm_tool_content(normalized)
            is_error = not normalized["ok"]

            if is_error:
                logger.warning(f"Tool {tool_name} returned error: {normalized.get('error')}")
            else:
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.debug(f"Tool {tool_name} result: {content_preview}")

            block: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
            }
            if is_error:
                block["is_error"] = True
            tool_results.append({"role": "user", "content": [block]})
            records[str(tool_id)] = {
                "name": tool_name,
                "parameters": arguments,
                "normalized": normalized,
                "llm_content": content,
            }

        logger.info(f"Completed {len(tool_results)} tool call results")
        return tool_results, records

    async def chat(self, message: str, user_info: Optional[Dict[str, Any]] = None) -> Union[str, AgentResponse]:
        """
        Send a message to Claude and get a response.

        Args:
            message: The user's message
            user_info: Optional dict containing info about the user's current state

        Returns:
            Either a string response (for backward compatibility) or a structured AgentResponse
            containing the message, tool_calls made, and optional thinking
        """
        # Format the user message with user_info if provided
        formatted_message = self.format_user_message(message, user_info)

        logger.info("Sending message to Claude API")
        logger.debug(f"Message length: {len(formatted_message)} chars")

        # Add the user message to the conversation history
        self.conversation_history.append({"role": "user", "content": formatted_message})

        # Prepare the messages for the API call - exclude system message from the conversation history
        # because Anthropic API requires system prompt as a separate parameter
        messages = []
        for msg in self.conversation_history:
            if msg["role"] != "system" and msg.get("content"):  # Ensure content is not empty
                messages.append(msg)

        # Always enable tools regardless of message content
        use_tools = True
        # Note: Previous code disabled tools for file-related operations, but this was causing issues
        # with the file_tools_demo and other demos that need to use file tools

        # Prepare tools if needed
        tools = None
        if use_tools:
            tools = self._prepare_tools()

        # Initialize the structured response
        processed_tool_calls: List[AgentToolCall] = []
        thinking = None  # Claude doesn't directly expose thinking, but we could add it in the future

        try:
            # Make the API call
            max_tokens_value = self._resolve_max_tokens(4096)
            api_params = {
                "model": self.model if self.model else "claude-3-5-sonnet-latest",
                "max_tokens": max_tokens_value,
                "temperature": self.temperature,
                "system": self.system_prompt,  # System prompt as a separate parameter
            }

            # Add properly typed messages
            typed_messages = []
            for msg in messages:
                if isinstance(msg["content"], str):
                    typed_messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    # Handle content that's not a string (e.g., structured content)
                    typed_messages.append(msg)

            api_params["messages"] = typed_messages

            # Only include tools parameter if we have tools registered and are using tools
            if tools and use_tools:
                api_params["tools"] = tools

            # Log API call
            logger.debug(f"Calling Claude API with {len(typed_messages)} messages")
            logger.debug(f"Using model: {api_params['model']}")
            if tools:
                logger.debug(f"Using {len(tools)} tools")

            # Make the API call
            logger.debug("Initiating API call to Claude")
            response = await self.client.messages.create(**api_params)  # type: ignore
            logger.info("Received response from Claude API")

            # Single completion + at most one tool-exec round.
            # Multi-step continuation belongs to run_agent_interactive (harness).
            response_text = ""
            has_tools = response.content and any(
                block.type == "tool_use" for block in response.content
            )

            if not has_tools:
                response_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                logger.debug(f"Response text length: {len(response_text)} chars")
                self.conversation_history.append(
                    {"role": "assistant", "content": response.content}
                )
            else:
                logger.info("Claude tool round (single; harness continues)")

                assistant_content = []
                for block in response.content:
                    if hasattr(block, "text") and block.text is not None:
                        assistant_content.append({"type": "text", "text": block.text})  # type: ignore
                    elif block.type == "tool_use":
                        assistant_content.append(  # type: ignore
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )

                self.conversation_history.append(
                    {"role": "assistant", "content": assistant_content}
                )

                tool_calls = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append(
                            {"name": block.name, "id": block.id, "input": block.input}
                        )

                logger.info(f"Extracted {len(tool_calls)} tool calls from response")
                tool_results, records = self._execute_tool_calls_with_records(tool_calls)

                for tool_call in tool_calls:
                    record = records.get(str(tool_call["id"]))
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

                response_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
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
            error_msg = f"Error: Authentication failed. Please check your Anthropic API key. Details: {str(e)}"
            logger.error(f"Authentication error: {str(e)}")
            return {
                "message": error_msg,
                "tool_calls": [],
                "thinking": None
            }
        except BadRequestError as e:
            # Provide more detailed information about the bad request
            request_info = ""
            if hasattr(e, "request"):
                request_info = f"\nRequest information: {e.request}"
            error_msg = f"Error: Bad request to the Anthropic API. Details: {str(e)}{request_info}"
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
            error_msg = f"Error: Anthropic API error. Details: {str(e)}"
            logger.error(f"API error: {str(e)}")
            return {
                "message": error_msg,
                "tool_calls": [],
                "thinking": None
            }
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred. Details: {type(e).__name__}: {str(e)}"
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
        Get structured JSON output from Claude based on the provided schema.
        Uses Claude's tool calling capabilities to enforce the output structure.

        Args:
            prompt: The prompt describing what structured data to generate
            schema: JSON schema defining the structure of the response
            model: Optional alternative Claude model to use for this request

        Returns:
            Dictionary containing the structured response that conforms to the schema
        """
        logger.info("Getting structured output from Claude")

        # Use specified model or default to the agent's model
        model_to_use = model or self.model

        # Create a temporary tool that defines the expected output structure
        structured_output_tool = {
            "name": "generate_structured_output",
            "description": "Generate a structured output response based on the provided schema",
            "input_schema": schema
        }

        try:
            # Create a message to the Claude API
            # Use max_tokens from extra_kwargs if available, otherwise default to 2000 for structured data
            max_tokens_value = self._resolve_max_tokens(2000)

            response = await self.client.messages.create(
                model=model_to_use,
                max_tokens=max_tokens_value,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                tools=[structured_output_tool],
                temperature=0
            )

            # Process the response content
            if response.content:
                for content in response.content:
                    if content.type == "tool_use":
                        # Extract the structured data from the tool call
                        tool_data = content.tool_use.input
                        logger.debug(f"Received structured data: {json.dumps(tool_data)[:100]}...")
                        return tool_data
                    elif content.type == "text":
                        # If we got text content instead of a tool call, try to parse JSON from it
                        try:
                            # Look for JSON-like content in the text
                            import re
                            json_match = re.search(r'\{.*\}', content.text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                structured_data = json.loads(json_str)
                                logger.debug(f"Parsed JSON from text response: {json.dumps(structured_data)[:100]}...")
                                return structured_data
                        except (json.JSONDecodeError, AttributeError) as e:
                            logger.warning(f"Could not parse JSON from text response: {str(e)}")

            # If no valid response found, log an error and return empty dict
            logger.error("No valid structured output found in Claude's response")
            return {}

        except Exception as e:
            logger.error(f"Error getting structured output from Claude: {str(e)}")
            return {}

    def _permission_request_callback(self, permission_request: PermissionRequest) -> PermissionStatus:
        """
        Implementation of permission request callback for Claude agent.

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
        Query the Claude model about one or more images.

        Args:
            image_paths: List of paths to local image files
            query: The query/question about the image(s)

        Returns:
            The model's response about the image(s)
        """
        import os
        import base64
        import mimetypes

        logger.info(f"Processing image query with {len(image_paths)} images")

        # Validate image paths
        for path in image_paths:
            if not os.path.exists(path):
                error_msg = f"Image file not found: {path}"
                logger.error(error_msg)
                return error_msg

        # Prepare images for the API
        content_blocks = [
            {"type": "text", "text": query}
        ]

        for path in image_paths:
            try:
                with open(path, "rb") as image_file:
                    # Get MIME type for the image
                    mime_type, _ = mimetypes.guess_type(path)
                    if not mime_type or not mime_type.startswith('image/'):
                        # Default to jpeg if type cannot be determined
                        mime_type = "image/jpeg"

                    # Read and encode the image
                    image_data = image_file.read()
                    encoded_image = base64.b64encode(image_data).decode('utf-8')

                    # Add image to content blocks
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": encoded_image
                        }
                    })
                    logger.debug(f"Processed image: {path} ({mime_type})")
            except Exception as e:
                error_msg = f"Error processing image {path}: {str(e)}"
                logger.error(error_msg)
                return error_msg

        try:
            # Set up the system prompt for image analysis

            if not self.system_prompt:
                image_system_prompt = "You are Claude, an AI assistant that can analyze and describe images. Provide detailed and accurate information about the images based on the user's query."
            else:
                image_system_prompt = self.system_prompt

            # Call the Claude API
            logger.debug(f"Calling Claude API for image analysis with model: {self.model}")

            # Use max_tokens from extra_kwargs if available, otherwise default to 1024 for vision
            max_tokens_value = self._resolve_max_tokens(1024)

            response = await self.client.messages.create(
                model=self.model,
                system=image_system_prompt,
                max_tokens=max_tokens_value,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": content_blocks
                    }
                ]
            )

            # Extract and return the assistant's response
            if response.content and len(response.content) > 0:
                result = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        result += content_block.text

                logger.info("Successfully processed image query")
                return result
            else:
                error_msg = "No response received from Claude API"
                logger.error(error_msg)
                return error_msg

        except BadRequestError as e:
            error_msg = f"Bad request to Claude API: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except RateLimitError as e:
            error_msg = f"Rate limit exceeded: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except APIError as e:
            error_msg = f"Claude API error: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error processing image query: {str(e)}"
            logger.error(error_msg)
            return error_msg
