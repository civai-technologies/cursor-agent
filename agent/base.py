import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .permissions import PermissionManager, PermissionOptions, PermissionRequest, PermissionStatus


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
    ):
        """
        Initialize the agent.

        Args:
            api_key: API key for the model provider. If not provided, will attempt to load from environment.
            model: Model to use. If not provided, will use the default model.
            permission_options: Configuration options for permissions
            permission_callback: Optional callback for handling permission requests
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

    @abstractmethod
    def _generate_system_prompt(self) -> str:
        """
        Generate the system prompt that defines the agent's capabilities and behavior.

        Returns:
            The system prompt as a string
        """
        pass

    @abstractmethod
    async def chat(self, message: str, user_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a message to the AI and get a response.

        Args:
            message: The user's message
            user_info: Optional dict containing info about the user's current state

        Returns:
            The AI's response
        """
        pass

    def register_tool(
        self, name: str, function: Callable, description: str, parameters: Dict[str, Any]
    ) -> None:
        """
        Register a function that can be called by the AI.

        Args:
            name: Name of the function
            function: The actual function to call
            description: Description of what the function does
            parameters: Dict describing the parameters the function takes
        """
        self.available_tools[name] = {
            "function": function,
            "schema": {"name": name, "description": description, "parameters": parameters},
        }

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
