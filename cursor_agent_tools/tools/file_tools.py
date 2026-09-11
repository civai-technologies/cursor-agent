import os
from typing import Any, Dict, Optional, Union
import json

from ..base import BaseAgent
from ..logger import get_logger

# Define exported functions
__all__ = [
    "read_file",
    "edit_file",
    "delete_file",
    "create_file",
    "list_directory"
]

# Initialize logger
logger = get_logger(__name__)


def read_file(
    target_file: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    should_read_entire_file: bool = False,
    agent: Optional[BaseAgent] = None,  # Optional agent reference for permissions
) -> Dict[str, Any]:
    """
    Read the contents of a file.

    Args:
        target_file: The path of the file to read
        offset: The line number to start reading from (1-indexed)
        limit: The number of lines to read
        should_read_entire_file: Whether to read the entire file
        agent: Reference to the agent instance for permission checks

    Returns:
        Dict containing the file contents
    """
    # No permission required for reading files
    try:
        # Handle the case where target_file is a dictionary with a path key
        if isinstance(target_file, dict) and "path" in target_file:
            file_path = target_file["path"]
        else:
            file_path = target_file

        logger.debug(f"Reading file: {file_path}")

        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return {"error": f"File {file_path} does not exist"}

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if should_read_entire_file:
            content = "".join(lines)
            logger.debug(f"Read entire file ({len(lines)} lines): {file_path}")
            return {"content": content, "total_lines": len(lines)}

        # Ensure offset and limit are valid
        if offset is None:
            offset = 1
        if limit is None:
            limit = 150  # Default to 150 lines

        # Convert to 0-indexed
        offset_idx = max(0, offset - 1)
        end_idx = min(len(lines), offset_idx + limit)

        # Extract the requested lines
        content_lines = lines[offset_idx:end_idx]
        content = "".join(content_lines)

        # Calculate summary
        summary = []
        if offset_idx > 0:
            summary.append(f"... {offset_idx} lines before ...")
        if end_idx < len(lines):
            summary.append(f"... {len(lines) - end_idx} lines after ...")

        # For tests that expect the end_line to be the actual line number,
        # ensure this correctly represents the last line we read
        end_line = offset + len(content_lines) - 1
        if end_line < offset:
            end_line = offset

        # If we read to the end of the file due to a small file size,
        # set end_line to the total number of lines
        if len(content_lines) > 0 and end_idx == len(lines):
            end_line = len(lines)

        logger.debug(f"Read file {file_path} from line {offset} to {end_line}")
        return {
            "content": content,
            "start_line": offset,
            "end_line": end_line,
            "summary": summary,
            "total_lines": len(lines),
        }

    except Exception as e:
        logger.error(f"Error reading file {target_file}: {str(e)}")
        return {"error": str(e)}


def edit_file(
    target_file: str,
    instructions: str,
    code_edit: Optional[Union[Dict[str, str], str]] = None,
    code_replace: Optional[str] = None,
    agent: Optional[BaseAgent] = None,  # Optional agent reference for permissions
) -> Dict[str, Any]:
    """
    Edit a file according to the provided instructions and code changes.

    Args:
        target_file: Path to the file to edit
        instructions: Instructions describing the edit
        code_edit: Line-based edit as a JSON dictionary or string with line ranges as keys:
                  {"1-5": "new content", "10-12": "more content"}
        code_replace: Complete replacement content for the file (if code_edit not provided)
        agent: Reference to the agent instance for permission checks

    Returns:
        Dict containing the status of the edit operation
    """
    try:
        logger.info(f"Editing file: {target_file}")
        logger.debug(f"Edit instructions: {instructions}")

        # Validate that we have either code_edit or code_replace, but not both
        if code_edit is None and code_replace is None:
            logger.error("Either code_edit or code_replace must be provided")
            return {"status": "error", "message": "Either code_edit or code_replace must be provided"}

        if code_edit is not None and code_replace is not None:
            logger.warning("Both code_edit and code_replace provided, using code_edit")

        # Create preview for logging and permissions
        if code_edit is not None:
            if isinstance(code_edit, dict):
                edit_preview = str(code_edit)[:300] + ("..." if len(str(code_edit)) > 300 else "")
            else:
                edit_preview = str(code_edit)[:300] + ("..." if len(str(code_edit)) > 300 else "")
            logger.debug(f"Code edit preview: {edit_preview}")
        elif code_replace is not None:
            replace_preview = code_replace[:300] + ("..." if len(code_replace) > 300 else "")
            logger.debug(f"Code replace preview: {replace_preview}")

        # Request permission if agent is provided
        if agent:
            operation_details = {
                "target_file": target_file,
                "instructions": instructions,
            }

            if code_edit is not None:
                operation_details["edit_preview"] = edit_preview if 'edit_preview' in locals() else str(code_edit)[:300]
            elif code_replace is not None:
                operation_details["replace_preview"] = replace_preview if 'replace_preview' in locals() else code_replace[:300]

            if not agent.request_permission("edit_file", operation_details):
                logger.warning(f"Permission denied to edit file: {target_file}")
                return {"status": "error", "message": "Permission denied to edit file"}

        # Check if file exists
        if not os.path.exists(target_file):
            logger.warning(f"File does not exist: {target_file}")
            return {"status": "error", "message": f"File {target_file} does not exist"}

        # Read the original content
        with open(target_file, "r") as f:
            original_content = f.read()

        # Apply the edit based on which parameter was provided
        if code_edit is not None:
            # Handle line-based edits
            if isinstance(code_edit, dict):
                # Convert all keys to strings
                line_edits = {str(k): v for k, v in code_edit.items()}
                edited_content = apply_line_based_edit(original_content, line_edits)
            elif isinstance(code_edit, str):
                # Try to parse as JSON if it looks like a JSON string
                code_edit = code_edit.strip()
                if code_edit.startswith('{') and code_edit.endswith('}'):
                    try:
                        line_edits = json.loads(code_edit)
                        if isinstance(line_edits, dict):
                            # Convert all keys to strings
                            line_edits_str = {str(k): v for k, v in line_edits.items()}
                            edited_content = apply_line_based_edit(original_content, line_edits_str)
                        else:
                            logger.error(f"Invalid JSON format for code_edit. Expected dict, got {type(line_edits)}")
                            return {"status": "error", "message": "Invalid JSON format for code_edit"}
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse code_edit as JSON: {str(e)}")
                        return {"status": "error", "message": f"Failed to parse code_edit as JSON: {str(e)}"}
                else:
                    # If it doesn't look like JSON, treat it as a complete replacement
                    logger.warning("code_edit doesn't appear to be JSON, treating as complete replacement")
                    edited_content = code_edit
            else:
                logger.error(f"Invalid type for code_edit: {type(code_edit)}")
                return {"status": "error", "message": f"Invalid type for code_edit: {type(code_edit)}"}
        else:
            # Use code_replace for complete file replacement
            edited_content = code_replace if code_replace is not None else ""

        # Write the edited content back to the file
        with open(target_file, "w") as f:
            f.write(edited_content)

        logger.info(f"Successfully edited file: {target_file}")
        return {"status": "success", "message": f"Successfully edited {target_file}"}

    except Exception as e:
        logger.error(f"Error editing file {target_file}: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


def delete_file(
    target_file: str,
    agent: Optional[BaseAgent] = None,  # Optional agent reference for permissions
) -> Dict[str, Any]:
    """
    Delete a file at the specified path.

    Args:
        target_file: The path of the file to delete
        agent: Reference to the agent instance for permission checks

    Returns:
        Dict containing the status of the deletion
    """
    try:
        logger.info(f"Deleting file: {target_file}")

        # Request permission if agent is provided
        if agent:
            operation_details = {
                "target_file": target_file,
                "operation": "delete",
            }

            if not agent.request_permission("delete_file", operation_details):
                logger.warning(f"Permission denied to delete file: {target_file}")
                return {"status": "error", "message": "Permission denied to delete file"}

        if not os.path.exists(target_file):
            logger.warning(f"File does not exist: {target_file}")
            return {"status": "error", "message": f"File {target_file} does not exist"}

        os.remove(target_file)
        logger.info(f"Successfully deleted file: {target_file}")
        return {"status": "success", "message": f"Deleted file {target_file}"}

    except Exception as e:
        logger.error(f"Error deleting file {target_file}: {str(e)}")
        return {"error": str(e)}


def create_file(
    file_path: str,
    content: str,
    agent: Optional[BaseAgent] = None,  # Optional agent reference for permissions
) -> Dict[str, Any]:
    """
    Create a new file with the given content.

    Args:
        file_path: Path where the file should be created
        content: Content to write to the file
        agent: Reference to the agent instance for permission checks

    Returns:
        Dict containing the status of the creation
    """
    try:
        logger.info(f"Creating file: {file_path}")

        # Request permission if agent is provided
        if agent:
            operation_details = {
                "file_path": file_path,
                "content_preview": content[:100] + ("..." if len(content) > 100 else ""),
            }

            if not agent.request_permission("create_file", operation_details):
                logger.warning(f"Permission denied to create file: {file_path}")
                return {"status": "error", "message": "Permission denied to create file"}

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Check if file already exists
        file_exists = os.path.exists(file_path)

        with open(file_path, "w") as f:
            f.write(content)

        if file_exists:
            logger.info(f"Updated existing file: {file_path}")
            return {"status": "success", "message": f"Updated file at {file_path}"}
        else:
            logger.info(f"Created new file: {file_path}")
            return {"status": "success", "message": f"Created file at {file_path}"}

    except Exception as e:
        logger.error(f"Error creating file {file_path}: {str(e)}")
        return {"error": str(e)}


def list_directory(
    path: str,
    agent: Optional[BaseAgent] = None,  # Optional agent reference for permissions
) -> Dict[str, Any]:
    """
    List the contents of a directory.

    Args:
        path: Absolute path of the directory to list. A relative value is
            resolved against the current working directory as a fallback.
        agent: Reference to the agent instance for permission checks

    Returns:
        Dict containing the directory contents
    """
    # No permission required for listing directories
    try:
        path = os.path.abspath(path)
        logger.debug(f"Listing directory: {path}")

        if not os.path.exists(path):
            logger.warning(f"Directory does not exist: {path}")
            return {"error": f"Directory {path} does not exist"}

        if not os.path.isdir(path):
            logger.warning(f"Not a directory: {path}")
            return {"error": f"{path} is not a directory"}

        # Get the directory contents
        contents = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            item_type = "dir" if os.path.isdir(item_path) else "file"
            item_size = os.path.getsize(item_path) if item_type == "file" else None

            contents.append({"name": item, "type": item_type, "size": item_size, "path": item_path})

        logger.debug(f"Listed {len(contents)} items in directory: {path}")
        return {"contents": contents}

    except Exception as e:
        logger.error(f"Error listing directory {path}: {str(e)}")
        return {"error": str(e)}


def apply_edit(original_content: str, code_edit: Any) -> str:
    """
    Parse and apply an edit to the original content.

    DEPRECATED: This function is maintained for backward compatibility only.
    New code should use edit_file with code_edit or code_replace parameters directly.

    Args:
        original_content: The original file content
        code_edit: The edit to apply, which should be:
                  - A dictionary with line ranges as keys and content as values
                    e.g., {"1-20": "new content", "30-35": "more content"}
                  - A JSON string representing such a dictionary
                  - Or complete replacement content as a string

    Returns:
        The content after applying the edit
    """
    logger.debug(f"apply_edit received code_edit of type: {type(code_edit)}")
    logger.warning("apply_edit is deprecated, prefer using edit_file with code_edit/code_replace")

    # Handle dictionary input
    if isinstance(code_edit, dict):
        # Ensure all keys are strings
        line_edits = {}
        for k, v in code_edit.items():
            if isinstance(k, slice):
                # Convert slice object to string format
                start = k.start or 1
                stop = k.stop or start
                k_str = f"{start}-{stop}"
                line_edits[k_str] = v
            else:
                # Just convert to string if not already
                line_edits[str(k)] = v

        logger.debug(f"Converted dictionary: {line_edits}")
        return apply_line_based_edit(original_content, line_edits)

    # Handle string input - try to parse as JSON
    if isinstance(code_edit, str):
        # Check if it looks like a JSON object
        code_edit = code_edit.strip()
        if code_edit.startswith('{') and code_edit.endswith('}'):
            try:
                # Attempt to parse as JSON
                line_edits = json.loads(code_edit)

                # Verify it's a dictionary and process it
                if isinstance(line_edits, dict):
                    # Convert all keys to strings
                    line_edits_str = {str(k): v for k, v in line_edits.items()}
                    logger.debug(f"Detected line-based edit with {len(line_edits_str)} ranges")
                    return apply_line_based_edit(original_content, line_edits_str)
                else:
                    logger.debug(f"JSON parsed but result is not a dictionary: {type(line_edits)}")
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse as JSON: {str(e)}")
                logger.debug(f"Content that failed to parse: {code_edit}")

    # If not a valid line-based edit, replace the entire content
    logger.debug("No valid line-based edit detected, replacing entire content")
    return str(code_edit)


def apply_line_based_edit(original_content: str, line_edits: Dict[str, str]) -> str:
    """
    Apply edits based on line ranges.

    Args:
        original_content: The original file content
        line_edits: Dictionary with keys as line ranges (e.g., "1-5", "7-7")
                   and values as the new content for those ranges

    Returns:
        The content after applying the line-based edits
    """
    logger.debug(f"Applying line-based edit: {line_edits}")
    original_lines = original_content.splitlines()
    result_lines = original_lines.copy()

    try:
        # Sort the line ranges to process from bottom to top
        # This prevents changes to line numbers affecting subsequent edits
        line_ranges = sorted(
            line_edits.keys(),
            key=lambda x: [int(n) if n.isdigit() else 0 for n in x.replace('-', ',').split(',')],
            reverse=True
        )

        logger.debug(f"Sorted line ranges: {line_ranges}")

        for line_range in line_ranges:
            try:
                # Parse the line range
                if "-" in line_range:
                    parts = line_range.split("-")
                    start = int(parts[0])
                    end = int(parts[1])
                else:
                    start = end = int(line_range)

                # Convert to 0-indexed
                start_idx = start - 1
                end_idx = end - 1

                # Validate indices
                if start_idx < 0:
                    logger.warning(f"Invalid start index {start_idx+1}, adjusting to 1")
                    start_idx = 0

                if end_idx >= len(original_lines):
                    logger.warning(f"Invalid end index {end_idx+1}, adjusting to {len(original_lines)}")
                    end_idx = len(original_lines) - 1

                if start_idx > end_idx:
                    logger.warning(f"Invalid line range: {line_range} (start > end)")
                    continue

                # Get the new content for this range
                new_content = line_edits[line_range]
                new_lines = new_content.splitlines()

                logger.debug(f"Replacing lines {start_idx+1}-{end_idx+1} with {len(new_lines)} new lines")
                logger.debug(f"Original lines: {original_lines[start_idx:end_idx+1]}")
                logger.debug(f"New lines: {new_lines}")

                # Replace the specified lines with the new content
                result_lines[start_idx:end_idx + 1] = new_lines

                logger.debug(f"Applied edit to lines {line_range}")

            except (ValueError, IndexError) as e:
                logger.error(f"Error applying edit to line range {line_range}: {str(e)}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")

        return "\n".join(result_lines)

    except Exception as e:
        logger.error(f"Error in apply_line_based_edit: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        # Return original content on error to avoid data loss
        return original_content
