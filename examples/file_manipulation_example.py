#!/usr/bin/env python3
"""
Example showing how to use the cursor agent for file manipulation.
"""

import os
import sys
import traceback
import asyncio
import platform
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Add the parent directory to the path so we can import the agent package
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cursor_agent_tools import ClaudeAgent
from cursor_agent_tools import OpenAIAgent
from cursor_agent_tools.tools.file_tools import read_file, edit_file, list_directory
from examples.utils import (
    Colors, print_error, print_system_message, 
    print_user_query, print_assistant_response, 
    print_info, print_separator
)

# Load environment variables from .env file
load_dotenv()

async def run_file_manipulation_example():
    """Run the file manipulation example."""
    try:
        print_system_message("Initializing agent...")
        
        # Get API keys
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        # Choose model based on available API keys
        if anthropic_key:
            agent = ClaudeAgent(
                api_key=anthropic_key,
                model="claude-3-5-sonnet-latest"
            )
            print_system_message("Using Claude model")
        elif openai_key:
            agent = OpenAIAgent(
                api_key=openai_key,
                model="gpt-4o"
            )
            print_system_message("Using GPT-4o model")
        else:
            print_error("No API keys found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variables.")
            return
        
        # Register file manipulation tools with the agent
        agent.register_tool(read_file)
        agent.register_tool(edit_file)
        agent.register_tool(list_directory)
        
        # Create a basic user info context
        user_info = {
            "os": platform.system(),
            "workspace_path": os.getcwd(),
        }
        
        # Create a temporary directory for file manipulation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sample file in the temp directory
            sample_file = os.path.join(temp_dir, "sample.txt")
            with open(sample_file, "w") as f:
                f.write("This is a sample text file.\nIt has multiple lines.\nThe agent will modify this file.")
            
            print_system_message(f"Created sample file at: {sample_file}")
            
            # Update user info with the temp directory
            user_info["workspace_path"] = temp_dir
            
            # Demo query for file manipulation
            query = f"Read the content of sample.txt, then modify it to add a fourth line saying 'This line was added by the agent.'"
            
            print_system_message("Sending query to agent...")
            print_user_query(query)
            
            # Get the response from the agent
            response = await agent.chat(query, user_info)
            
            # Process the structured response
            if isinstance(response, dict):
                # Print the agent's message
                print_assistant_response(response["message"])
                
                # Show tool usage if present
                if response.get("tool_calls"):
                    print_info(f"\nAgent used {len(response['tool_calls'])} tool calls:")
                    for i, call in enumerate(response['tool_calls'], 1):
                        print_info(f"\n{i}. Tool: {call['name']}")
                        if call["name"] == "read_file" and "target_file" in call["parameters"]:
                            print_info(f"   Read file: {call['parameters']['target_file']}")
                        elif call["name"] == "edit_file" and "target_file" in call["parameters"]:
                            print_info(f"   Edited file: {call['parameters']['target_file']}")
                        elif call["name"] == "list_directory" and "path" in call["parameters"]:
                            print_info(f"   Listed directory: {call['parameters']['path']}")
            else:
                # Backward compatibility
                print_assistant_response(response)
                
            # Show the final content of the file
            if os.path.exists(sample_file):
                print_separator()
                print_system_message("Final content of sample.txt:")
                with open(sample_file, "r") as f:
                    content = f.read()
                print(content)
                
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        traceback.print_exc()

async def main():
    """Main entry point for the file manipulation example."""
    
    try:
        print_separator()
        print_system_message("FILE MANIPULATION EXAMPLE")
        print_separator()
        
        # Run the file manipulation example
        await run_file_manipulation_example()
        
        print_separator()
        print_system_message("FILE MANIPULATION EXAMPLE COMPLETED")
        
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        traceback.print_exc()
        
if __name__ == "__main__":
    asyncio.run(main())
