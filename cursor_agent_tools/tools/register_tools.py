"""
Utility module for registering agent tools with permission handling.
"""

from typing import Any, Dict, List
import asyncio
import concurrent.futures
import os

from ..logger import get_logger
from ..schemas import TERMINATE_TEXT_SENTINEL, TERMINATE_TOOL_NAME
from . import (
    file_tools,
    search_tools,
    system_tools,
    image_tools,
)

# Define exported functions
__all__ = ["register_default_tools"]

# Initialize logger
logger = get_logger(__name__)


def register_default_tools(agent: Any) -> None:
    """
    Register all available tools with the provided agent.

    The agent reference is automatically injected into the tool functions
    by the agent framework for permission handling.

    Args:
        agent: The agent instance to register tools with
    """
    logger.info("Registering default tools for agent")

    # File tools with permission checks
    agent.register_tool(
        "read_file",
        lambda target_file, offset=None, limit=None, should_read_entire_file=None: file_tools.read_file(
            target_file, offset, limit, should_read_entire_file, agent
        ),
        "Read the contents of a file.",
        {
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": "Absolute path of the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from (1-indexed)",
                },
                "limit": {"type": "integer", "description": "The number of lines to read"},
                "should_read_entire_file": {
                    "type": "boolean",
                    "description": "Whether to read the entire file",
                },
            },
            "required": ["target_file"],
        },
        arg_aliases={"path": "target_file", "file_path": "target_file"},
    )
    logger.debug("Registered tool: read_file")

    agent.register_tool(
        "edit_file",
        lambda target_file, instructions, code_edit=None, code_replace=None: file_tools.edit_file(
            target_file, instructions, code_edit, code_replace, agent
        ),
        "Edit a file in the codebase.",
        {
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": "Absolute path of the file to modify",
                },
                "instructions": {
                    "type": "string",
                    "description": "A single sentence instruction describing the edit",
                },
                "code_edit": {
                    # OpenAI-strict: single type only (no union/nullable type lists).
                    # Optional = omitted from required; implementation still accepts dict or JSON string.
                    "type": "object",
                    "description": (
                        "Line-based edit with line ranges as keys (e.g. \"1-5\") and new content as values. "
                        "Omit this field if using code_replace instead."
                    ),
                    "additionalProperties": {
                        "type": "string",
                        "description": "New content for the specified line range",
                    },
                },
                "code_replace": {
                    "type": "string",
                    "description": (
                        "Complete replacement content for the file "
                        "(use instead of code_edit for full file replacement). Omit if using code_edit."
                    ),
                },
            },
            "required": ["target_file", "instructions"],
        },
        arg_aliases={"path": "target_file", "file_path": "target_file"},
    )
    logger.debug("Registered tool: edit_file")

    agent.register_tool(
        "delete_file",
        lambda target_file: file_tools.delete_file(target_file, agent),
        "Delete a file at the specified path.",
        {
            "type": "object",
            "properties": {
                "target_file": {
                    "type": "string",
                    "description": "Absolute path of the file to delete",
                },
            },
            "required": ["target_file"],
        },
        arg_aliases={"path": "target_file", "file_path": "target_file"},
    )
    logger.debug("Registered tool: delete_file")

    agent.register_tool(
        "create_file",
        lambda file_path, content: file_tools.create_file(file_path, content, agent),
        "Create a new file with the given content.",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path where the file should be created",
                },
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["file_path", "content"],
        },
        arg_aliases={"path": "file_path", "target_file": "file_path"},
    )
    logger.debug("Registered tool: create_file")

    def _list_directory(path: str = "") -> Dict[str, Any]:
        """List a directory; empty path defaults to the agent workspace root."""
        resolved = path or getattr(agent, "workspace_root", "") or os.getcwd()
        return file_tools.list_directory(resolved, agent)

    agent.register_tool(
        "list_directory",
        _list_directory,
        "List the contents of a directory.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path of the directory to list. "
                        "Omit to list the working directory."
                    ),
                },
            },
            "required": [],
        },
        arg_aliases={
            "directory": "path",
            "dir": "path",
            "directory_path": "path",
            "target_directory": "path",
            "folder": "path",
            "file_path": "path",
            "target_file": "path",
        },
    )
    logger.debug("Registered tool: list_directory")

    # System tools
    agent.register_tool(
        "run_terminal_command",
        lambda command, explanation=None, is_background=False, require_user_approval=True: system_tools.run_terminal_command(
            command, explanation, is_background, require_user_approval, agent
        ),
        "Run a terminal command. IMPORTANT: Always use non-interactive flags that works for the command you are running (like --yes, -y, --no-interaction, yes | , --quiet, or equivalent). i.e git commit -m 'commit message' --no-interaction or yes | npx create-next-app@latest --no-interactive",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The terminal command to execute. MUST be non-interactive and should include all necessary flags to prevent or bypass prompts."},
                "explanation": {
                    "type": "string",
                    "description": "Explanation of why this command needs to be run",
                },
                "is_background": {
                    "type": "boolean",
                    "description": "Whether to run in the background",
                },
                "require_user_approval": {
                    "type": "boolean",
                    "description": "Whether user approval is required",
                },
            },
            "required": ["command"],
        },
    )
    logger.debug("Registered tool: run_terminal_command")

    # Search tools
    agent.register_tool(
        "codebase_search",
        lambda query, target_directories=None, explanation=None: search_tools.codebase_search(
            query, target_directories, explanation, agent
        ),
        "Search the codebase using semantic search.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant code",
                },
                "target_directories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Directories to search over",
                },
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used",
                },
            },
            "required": ["query"],
        },
    )
    logger.debug("Registered tool: codebase_search")

    agent.register_tool(
        "grep_search",
        lambda query, explanation=None, case_sensitive=False, include_pattern=None, exclude_pattern=None: search_tools.grep_search(
            query, explanation, case_sensitive, include_pattern, exclude_pattern, agent
        ),
        "Fast text-based search using regex patterns.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The regex pattern to search for"},
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case sensitive",
                },
                "include_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to include",
                },
                "exclude_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files to exclude",
                },
            },
            "required": ["query"],
        },
    )
    logger.debug("Registered tool: grep_search")

    agent.register_tool(
        "file_search",
        lambda query, explanation=None: search_tools.file_search(query, explanation, agent),
        "Fast file search based on fuzzy matching against file path.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Fuzzy filename to search for"},
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used",
                },
            },
            "required": ["query"],
        },
    )
    logger.debug("Registered tool: file_search")

    # Web search
    agent.register_tool(
        "web_search",
        lambda search_term, explanation=None, force=False, objective=None, max_results=5: search_tools.web_search(
            search_term, explanation, force, objective, max_results, agent
        ),
        "Search the web for information.",
        {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "The search term to look up on the web",
                },
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this search is being performed",
                },
                "force": {
                    "type": "boolean",
                    "description": "Force internet access even if not required",
                },
                "objective": {
                    "type": "string",
                    "description": "User objective to determine if up-to-date data is needed",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 3)",
                },
            },
            "required": ["search_term"],
        },
    )
    logger.debug("Registered tool: web_search")

    # Register trend search tool
    agent.register_tool(
        "trend_search",
        lambda query, explanation=None, country_code="US", days=7, max_results=3, lookback_hours=48: asyncio.run(search_tools.trend_search(
            query=query,
            explanation=explanation,
            country_code=country_code,
            days=days,
            max_results=max_results,
            lookback_hours=lookback_hours,
            agent=agent
        )),
        "Search for trending topics related to a query",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search trends for"
                },
                "explanation": {
                    "type": "string",
                    "description": "Optional explanation of why this search is being performed"
                },
                "country_code": {
                    "type": "string",
                    "description": "Country code for trends (default: US)"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back for trends (default: 7)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of trends to return (default: 3)"
                },
                "lookback_hours": {
                    "type": "integer",
                    "description": "Number of hours to look back for Google Trends data (default: 48)"
                }
            },
            "required": ["query"]
        }
    )
    logger.debug("Registered tool: trend_search")

    def _sync_query_images(query: str, image_paths: List[str]) -> Dict[str, Any]:
        """Run async query_images from sync tool dispatch (agent chat is async)."""
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if not in_loop:
            return asyncio.run(image_tools.query_images(query, image_paths, agent))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, image_tools.query_images(query, image_paths, agent)
            ).result()

    # Image query tool
    agent.register_tool(
        "query_images",
        _sync_query_images,
        "Query an AI model about one or more images.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or query about the image(s)",
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of local paths to image files to analyze",
                },
            },
            "required": ["query", "image_paths"],
        },
    )
    logger.debug("Registered tool: query_images")

    # Explicit session/turn exit (model decides to stop — Nova-style terminate_session)
    def _terminate_agent_process(reason: str = "done") -> Dict[str, Any]:
        msg = f"{TERMINATE_TEXT_SENTINEL} {reason}".strip()
        return {"output": msg, "error": None}

    agent.register_tool(
        TERMINATE_TOOL_NAME,
        _terminate_agent_process,
        (
            "End this agent session/turn when the task is finished or no further tool "
            "work is needed. Prefer this over continuing to plan in text."
        ),
        {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short reason for ending (e.g. done, blocked, partial)",
                },
            },
            "required": [],
        },
    )
    logger.debug("Registered tool: terminate_agent_process")

    logger.info(f"Successfully registered {len(agent.available_tools)} tools")
