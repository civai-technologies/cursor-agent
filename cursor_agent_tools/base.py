"""Base agent module for handling agent operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union, TypedDict
import json

from .logger import get_logger
from .permissions import PermissionManager, PermissionOptions, PermissionRequest, PermissionStatus
from .schemas import (
    ToolResult,
    build_tool_event_payload,
    normalize_tool_result,
)


# Initialize logger
logger = get_logger(__name__)


class ToolCall(TypedDict):
    name: str
    parameters: Dict[str, Any]


class ToolResponse(TypedDict):
    output: str
    error: Optional[str]


class AgentToolCall(TypedDict):
    """Tool call made by an agent. Producers dual-emit ``result`` (legacy) and ``output``."""

    name: str
    parameters: Dict[str, Any]
    result: Optional[str]
    output: str
    error: Optional[str]
    thinking: Optional[str]


class _AgentResponseRequired(TypedDict):
    message: str
    tool_calls: List[AgentToolCall]
    thinking: Optional[str]


class AgentResponse(_AgentResponseRequired, total=False):
    """Structured agent response. Core fields are required; extras are additive."""

    primary_tool_call: AgentToolCall


class BaseAgent(ABC):
    """
    Base abstract class for AI agents that use function calling capabilities.
    This defines the common interface for all agents regardless of the underlying provider.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        permission_options: Optional[PermissionOptions] = None,
        permission_callback: Optional[Callable[[PermissionRequest], PermissionStatus]] = None,
        default_tool_timeout: int = 300,
    ):
        """
        Initialize the agent.

        Args:
            api_key: API key for the model provider. If not provided, will attempt to load from environment.
            model: Model to use. If not provided, will use the default model.
            permission_options: Configuration options for permissions
            permission_callback: Optional callback for handling permission requests
            default_tool_timeout: Default timeout for tool calls in seconds (default: 300s)
        """
        self.api_key: Optional[str] = api_key
        self.model: Optional[str] = model
        self.conversation_history: List[Dict[str, Any]] = []
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.system_prompt: str = self._generate_system_prompt()

        # Initialize permission manager with options and optional callback
        self.permission_manager = PermissionManager(
            options=permission_options or PermissionOptions(),
            callback=permission_callback
        )

        # Default timeout for tool executions
        self.default_tool_timeout = default_tool_timeout

        logger.debug(f"Initialized {self.__class__.__name__} with default tool timeout: {default_tool_timeout}s")

    @abstractmethod
    def _generate_system_prompt(self) -> str:
        """
        Generate the system prompt that defines the agent's capabilities and behavior.

        Returns:
            The system prompt as a string
        """
        pass

    @abstractmethod
    async def chat(self, message: str, user_info: Optional[Dict[str, Any]] = None) -> Union[str, AgentResponse]:
        """
        Send a message to the AI and get a response.

        Args:
            message: The user's message
            user_info: Optional dict containing info about the user's current state

        Returns:
            Either a string response (for backward compatibility) or a structured AgentResponse
            containing the message, tool_calls made, and optional thinking
        """
        pass

    @abstractmethod
    async def query_image(self, image_paths: List[str], query: str) -> str:
        """
        Query an LLM about one or more images.

        Args:
            image_paths: List of paths to local image files
            query: The query/question about the image(s)

        Returns:
            The model's response to the query about the image(s)
        """
        pass

    @abstractmethod
    async def get_structured_output(self, prompt: str, schema: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get structured JSON output from the agent based on the provided schema.

        Args:
            prompt: The prompt describing what structured data to generate
            schema: JSON schema defining the structure of the response
            model: Optional alternative model to use for this request

        Returns:
            Dictionary containing the structured response that conforms to the schema
        """
        pass

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any],
        arg_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Register a function that can be called by the AI.

        Args:
            name: Name of the function
            function: The actual function to call
            description: Description of what the function does
            parameters: Dict describing the parameters the function takes
            arg_aliases: Optional mapping of alternate argument keys a model
                may send to the canonical schema key (e.g. ``{"directory":
                "path"}``). Declared here so the tool registration owns its
                own contract; the executor applies them generically.
        """
        self.available_tools[name] = {
            "function": function,
            "schema": {"name": name, "description": description, "parameters": parameters},
            "arg_aliases": dict(arg_aliases or {}),
        }
        logger.debug(f"Registered tool: {name}")

    @abstractmethod
    def _prepare_tools(self) -> Any:
        """
        Format the registered tools into the format expected by the model's API.

        Returns:
            Tools in the format expected by the model
        """
        pass

    @abstractmethod
    def _execute_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """
        Execute the tool calls made by the AI.

        Args:
            tool_calls: Tool calls in the format provided by the specific model

        Returns:
            List of tool call results
        """
        pass

    def normalize_tool_call_arguments(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply registration-declared aliases, remap workspace paths, filter to schema.

        The tool registration owns the argument contract: canonical keys and
        their aliases live in ``register_tool(..., arg_aliases=)``. This
        pipeline is generic — it never hardcodes per-tool knowledge:

        1. alias keys → canonical schema keys (from registration metadata)
        2. optional host remap hook (``remap_tool_arguments``)
        3. filter to declared schema properties

        Required-argument validation happens in ``run_registered_tool`` so a
        missing arg becomes a structured ToolResult, never a TypeError.
        """
        if not isinstance(arguments, dict):
            return {}
        args: Dict[str, Any] = dict(arguments)

        tool_meta = (getattr(self, "available_tools", None) or {}).get(tool_name) or {}
        aliases = tool_meta.get("arg_aliases") or {}
        for src, dst in aliases.items():
            if src in args:
                if dst not in args or args.get(dst) in (None, ""):
                    args[dst] = args[src]
                if src != dst:
                    args.pop(src, None)

        remap = getattr(self, "remap_tool_arguments", None)
        if callable(remap):
            try:
                remapped = remap(args)
                if isinstance(remapped, dict):
                    args = remapped
            except Exception as remap_exc:
                logger.warning(f"Tool path remap failed: {remap_exc}")

        schema = tool_meta.get("schema") or {}
        params = schema.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        if isinstance(props, dict) and props:
            args = {k: v for k, v in args.items() if k in props}
        return args

    def _resolve_max_tokens(self, default: int) -> int:
        """Read max_tokens from extra_kwargs with a per-call default."""
        extra = getattr(self, "extra_kwargs", None)
        if isinstance(extra, dict):
            try:
                return int(extra.get("max_tokens", default))
            except (TypeError, ValueError):
                return default
        return default

    def _resolve_max_tool_rounds(self, default: int = 20) -> int:
        """Read max_tool_rounds from extra_kwargs (multi-round tool loop cap)."""
        extra = getattr(self, "extra_kwargs", None)
        if isinstance(extra, dict):
            try:
                return int(extra.get("max_tool_rounds", default))
            except (TypeError, ValueError):
                return default
        return default

    def notify_http_event(self, payload: Dict[str, Any]) -> None:
        """Optional host hook for outbound LLM HTTP round-trips (start/end)."""
        hook = getattr(self, "on_http_event", None)
        if not callable(hook):
            return
        try:
            hook(payload)
        except Exception as hook_exc:
            logger.debug(f"on_http_event hook failed: {hook_exc}")

    def run_registered_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """Execute a registered tool and emit a normalized ToolResult."""
        args = self.normalize_tool_call_arguments(tool_name, arguments)
        if tool_name not in self.available_tools:
            normalized = normalize_tool_result(
                {"error": f"Tool '{tool_name}' not found"}
            )
            self.notify_tool_event_normalized(tool_name, args, normalized)
            return normalized

        # Validate required schema args up front so a missing argument becomes
        # a structured ToolResult instead of a raw TypeError from the function.
        schema = self.available_tools[tool_name].get("schema") or {}
        params = schema.get("parameters") or {}
        required = params.get("required") if isinstance(params, dict) else None
        if isinstance(required, list):
            missing = [key for key in required if args.get(key) in (None, "")]
            if missing:
                normalized = normalize_tool_result(
                    {
                        "error": (
                            f"{tool_name}: missing required argument(s) "
                            + ", ".join(f"'{key}'" for key in missing)
                        )
                    }
                )
                self.notify_tool_event_normalized(tool_name, args, normalized)
                return normalized

        try:
            function = self.available_tools[tool_name]["function"]
            raw = function(**args)
            normalized = normalize_tool_result(raw)
            self.notify_tool_event_normalized(tool_name, args, normalized, raw=raw)
            return normalized
        except Exception as exc:
            normalized = normalize_tool_result({"error": str(exc)})
            self.notify_tool_event_normalized(tool_name, args, normalized)
            return normalized

    def notify_tool_event_normalized(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        normalized: ToolResult,
        *,
        raw: Any = None,
    ) -> None:
        """Emit dual-emit on_tool_event hook payload."""
        hook = getattr(self, "on_tool_event", None)
        if not callable(hook):
            return
        try:
            hook(build_tool_event_payload(tool_name, arguments, normalized, raw=raw))
        except Exception as hook_exc:
            logger.debug(f"on_tool_event hook failed: {hook_exc}")

    def notify_tool_event(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Optional host hook (e.g. pods Live progress) after each tool execute."""
        if result is not None:
            normalized = normalize_tool_result(result)
            if error is not None:
                normalized = {
                    "ok": False,
                    "output": normalized["output"],
                    "error": str(error),
                    "data": normalized.get("data"),
                }
        elif error is not None:
            normalized = normalize_tool_result({"error": error})
        else:
            normalized = normalize_tool_result({})
        self.notify_tool_event_normalized(tool_name, arguments, normalized, raw=result)

    def request_permission(
        self, operation_type: str, details: Dict[str, Any]
    ) -> bool:
        """
        Request permission for an operation.

        This method forwards the permission request to the permission manager.

        Args:
            operation_type: Type of operation ('create_file', 'edit_file', 'delete_file', 'run_terminal_command', etc.)
            details: Dictionary containing operation details

        Returns:
            True if permission is granted, False otherwise
        """
        return self.permission_manager.request_permission(operation_type, details)

    def _permission_request_callback(self, permission_request: PermissionRequest) -> PermissionStatus:
        """
        Default implementation of permission request callback.

        This method can be overridden by subclasses to provide
        appropriate user interaction for permission requests.

        Args:
            permission_request: The permission request object

        Returns:
            PermissionStatus indicating whether the request is granted, denied, or needs confirmation
        """
        # Default implementation prompts the user for confirmation
        print(f"\n🔒 Permission Request: {permission_request.operation}")
        print(f"Details: {json.dumps(permission_request.details, indent=2)}")

        while True:
            response = input("Allow this operation? (y/n): ").strip().lower()
            if response in ("y", "yes"):
                return PermissionStatus.GRANTED
            elif response in ("n", "no"):
                return PermissionStatus.DENIED
            else:
                print("Please enter 'y' or 'n'")

    def format_user_message(self, message: str, user_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Format the user message with user_info if provided.

        Args:
            message: The user's message
            user_info: Optional dict containing info about the user's current state

        Returns:
            Formatted message
        """
        if user_info:
            return f"<user_info>\n{json.dumps(user_info, indent=2)}\n</user_info>\n\n<user_query>\n{message}\n</user_query>"
        else:
            return f"<user_query>\n{message}\n</user_query>"

    def register_default_tools(self) -> None:
        """
        Register the default set of tools with the agent.

        This method imports and calls the register_default_tools function
        from the tools module, passing self as the agent.
        """
        # Import here to avoid circular imports
        from .tools.register_tools import register_default_tools
        register_default_tools(self)
        logger.info(f"Registered {len(self.available_tools)} default tools")
