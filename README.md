# Cursor Agent Tools

![License](https://img.shields.io/github/license/civai-technologies/cursor-agent)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-blueviolet)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT4-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20Models-orange)

A Python-based AI agent that replicates Cursor's coding assistant capabilities, enabling function calling, code generation, and intelligent coding assistance with Claude, OpenAI, and locally hosted Ollama models.

## 🌟 Features

This AI Agent implementation provides a comprehensive set of capabilities:

### Core Abilities

- **Model Flexibility**: Works with Claude (Anthropic), OpenAI models, and locally hosted Ollama models
- **Code Generation**: Generate complete, functional code based on natural language descriptions
- **Code Editing**: Make precise edits to existing code files
- **Code Analysis**: Review and analyze code for bugs, improvements, and optimizations
- **Function Calling**: Invoke registered tools and functions based on user requests
- **Conversational Context**: Maintain a conversation history for coherent back-and-forth interactions
- **Project-Aware Responses**: Consider project context when answering questions
- **Permission System**: Secure permission handling for file operations and command execution
- **Local Model Support**: Use open-source models hosted locally with Ollama

### Tool Functions

The agent supports a comprehensive set of tools:

- **File Operations**:
  - **read_file**: Read file contents with flexible line range control
  - **edit_file**: Make precise edits to files with clear instructions
    - Supports complete file replacement
    - Supports line-based editing with JSON dictionaries (e.g., `{"1-5": "new content", "10-12": "more content"}`)
  - **delete_file**: Remove files from the filesystem
  - **create_file**: Create new files with specified content
  - **list_dir**: List directory contents to understand project structure

- **Search Capabilities**:
  - **codebase_search**: Semantic search of codebases to find relevant code snippets
  - **grep_search**: Perform regex-based text search in files
  - **file_search**: Fuzzy search for files by name
  - **web_search**: Search the web for up-to-date information
  - **trend_search**: Search for trending topics and their latest developments

- **Image Analysis**:
  - **query_images**: Analyze and describe images using LLM vision capabilities

- **System Operations**:
  - **run_terminal_cmd**: Execute terminal commands with user approval

All tools are implemented with actual functionality and can be extended with custom tools as needed.

## 📚 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Permission System](#permission-system)
- [Contributing](#contributing)
- [API Documentation](#api-documentation)
- [Advanced Usage](#advanced-usage)
- [Limitations and Considerations](#limitations-and-considerations)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- API keys for Anthropic and/or OpenAI

### Via pip

```bash
pip install cursor-agent-tools
```

### From Source

```bash
git clone https://github.com/civai-technologies/cursor-agent.git
cd cursor-agent
pip install -e .  # Install in development mode

# Or with development dependencies
pip install -e ".[dev]"
```

### Environment Setup

Create a `.env` file in your project root (copy from `.env.example`):

```ini
# Environment (local, development, production)
ENVIRONMENT=local

# OpenAI configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.0

# Anthropic configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_API_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_TEMPERATURE=0.0

# Web search provider for web_search tool
# google (default) or you
WEB_SEARCH_PROVIDER=google

# Google Search API (required when WEB_SEARCH_PROVIDER=google)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# You.com Search API (optional key, supports free tier for Search API)
YDC_API_KEY=your_ydc_api_key_here

# Ollama configuration (for local models)
OLLAMA_HOST=http://localhost:11434
```

## ⚡ Quick Start

```python
import asyncio
from cursor_agent_tools import create_agent

async def main():
    # Create a Claude agent instance
    agent = create_agent(model='claude-3-5-sonnet-latest')
    
    # Chat with the agent
    response = await agent.chat("Create a Python function to calculate Fibonacci numbers")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 Usage Examples

### Chat with Different Models

```python
import asyncio
from cursor_agent_tools import create_agent

# Use Claude
claude_agent = create_agent(model='claude-3-5-sonnet-latest')
response = await claude_agent.chat("What's a good way to implement a cache in Python?")

# Use OpenAI
openai_agent = create_agent(model='gpt-4o')
response = await openai_agent.chat("What's a good way to implement a cache in Python?")

# Use Ollama (local open-source model)
ollama_agent = create_agent(model='ollama-llama3')
response = await ollama_agent.chat("What's a good way to implement a cache in Python?")
```

### Using Local Ollama Models

```python
import asyncio
from cursor_agent_tools import create_agent

async def main():
    # Create an agent with a local Ollama model
    # Models must be pulled via Ollama CLI first: ollama pull MODEL_NAME
    agent = create_agent(
        model='ollama-llama3',  # prefix with "ollama-" followed by model name
        host='http://localhost:11434',  # optional, this is the default
        temperature=0.3  # optional temperature setting
    )
    
    # Chat with the local model
    response = await agent.chat("Write a Python script to download YouTube videos")
    print(response)
    
    # Handle multimodal capabilities if model supports it
    # image_path = "/path/to/your/image.png"
    # image_response = await agent.query_image(
    #     image_paths=[image_path],
    #     query="What does this code screenshot show?"
    # )
    # print(image_response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Supported Ollama Models

The agent supports any model available in Ollama. Some popular options include:

- `ollama-llama3` - Meta's Llama 3 model
- `ollama-llama3.1` - Meta's Llama 3.1 model
- `ollama-mistral` - Mistral AI's model
- `ollama-gemma3` - Google's Gemma model
- `ollama-deepseek-r1` - DeepSeek's reasoning model
- `ollama-phi4` - Microsoft's Phi model
- `ollama-qwen2.5` - Qwen's latest model

For a complete list of available models, see [Ollama Library](https://ollama.com/library).

Note that tool calling and multimodal support depend on the capabilities of the specific model.

### Creating a Custom Agent with Custom System Prompt

```python
import asyncio
from cursor_agent_tools import create_agent

async def main():
    # Define a custom system prompt for a coding tutor
    custom_system_prompt = """
    You are an expert coding tutor specialized in helping beginners learn to code.
    When asked coding questions:
    1. First explain the concept in simple terms
    2. Always provide commented example code
    3. Suggest practice exercises
    4. Anticipate common mistakes and warn against them
    
    Be patient, encouraging, and avoid technical jargon unless you explain it.
    Focus on building good habits and understanding core principles.
    """
    
    # Create agent with custom system prompt
    coding_tutor = create_agent(
        model='claude-3-5-sonnet-latest',
        system_prompt=custom_system_prompt
    )
    
    # Example interaction with the custom agent
    student_question = "I'm confused about Python list comprehensions. Can you help me understand them?"
    
    response = await coding_tutor.chat(student_question)
    print(response)
    
    # Example using image analysis capabilities
    image_path = "/path/to/your/image.png"
    image_response = await coding_tutor.query_image(
        image_paths=[image_path],
        query="What does this code screenshot show and what issues do you see?"
    )
    print(image_response)
    
    # The response will follow the guidelines in the custom system prompt,
    # explaining list comprehensions in a beginner-friendly way with examples, 
    # practice exercises, and common pitfalls to avoid

if __name__ == "__main__":
    asyncio.run(main())
```

This example creates a specialized coding tutor agent with a custom personality and behavior guidelines. You can similarly create custom agents for various domains by crafting appropriate system prompts:

- Data analysis assistant (focusing on pandas/numpy)
- DevOps automation expert (for CI/CD and deployment scripts)
- Documentation writer (generating well-structured docs from code)
- Security code reviewer (identifying vulnerabilities in code)
- Algorithm optimization specialist (improving performance)

### Interactive Mode

```python
from cursor_agent_tools import run_agent_interactive
import asyncio

async def main():
    # Parameters:
    # - model: The model to use (e.g., 'claude-3-5-sonnet-latest', 'gpt-4o')
    # - initial_query: The task description
    # - max_iterations: Maximum number of steps (default 10)
    # - auto_continue: Whether to continue automatically without user input (default True)
    
    await run_agent_interactive(
        model='claude-3-5-sonnet-latest',
        initial_query='Create a simple web scraper that extracts headlines from a news website',
        max_iterations=15
        # auto_continue=True is the default - agent continues automatically
        # To disable automatic continuation, set auto_continue=False
    )

if __name__ == "__main__":
    asyncio.run(main())
```

The interactive mode is designed to automatically continue without user interaction unless:
1. The AI explicitly requests more information from the user
2. An error occurs that requires user decision
3. The `--auto` flag is used, which will prompt for user input after each step
4. The maximum tool call limit is reached, prompting for user confirmation to continue

When user input is requested or provided, the system intelligently incorporates this input into the conversation flow by:
1. Using the AI model itself to generate a contextually appropriate continuation prompt
2. Seamlessly integrating the user's input with the existing conversation history
3. Maintaining the natural flow of the implementation process

### Automatic User Input Detection

The interactive mode has built-in detection for when the agent is explicitly asking for user input. When the agent's response includes phrases like:
- "I need more information about..."
- "Could you provide more details on..."
- "Please let me know your preference for..."
- "What would you like me to do about..."

The system will automatically pause and wait for user input, even in auto-continue mode. This ensures that when the agent genuinely needs clarification or a decision from you, the conversation pauses appropriately.

### Tool Call Limit Confirmation

For safety and to prevent runaway automation, the agent tracks the total number of tool calls made during processing a single agent response. When this count reaches a certain threshold (default: 5), the agent will request user confirmation before making more changes. This is a safeguard that:
- Prevents unintended extensive changes to your codebase
- Gives you visibility into complex operations
- Allows you to review progress before continuing
- Provides an opportunity to redirect the agent if needed

Important details about how tool call tracking works:
- The counter tracks all tool calls within one logical iteration (one agent response)
- The counter is NOT reset until new user input is provided or a new query is started
- The counter persists even when the agent continues with the same iteration

When the limit is reached, you'll be prompted with:
```
The agent has made 5 tool calls in this iteration.
Would you like to continue allowing the agent to make more changes?
Continue? (y/n):
```

If you approve, the agent will:
- Continue making more tool calls in the current iteration
- Increase the limit by 5 for this iteration
- Ask for confirmation again when the new limit is reached
- Keep tracking the total number of tool calls (the counter is not reset)

If you deny, the agent will:
- Stop making additional tool calls for this iteration
- Complete the current iteration with the changes already made
- Continue to the next iteration when ready

This adaptive limit ensures the agent doesn't make too many changes without your approval, while still allowing complex tasks to be completed efficiently.

### Providing Project Context

```python
from cursor_agent_tools import create_agent

agent = create_agent(model='claude-3-5-sonnet-latest')

user_info = {
    "open_files": ["src/main.py", "src/utils.py"],
    "cursor_position": {"file": "src/main.py", "line": 42},
    "recent_files": ["src/config.py", "tests/test_main.py"],
    "os": "darwin",
    "workspace_path": "/Users/username/projects/myproject"
}

response = await agent.chat("Fix the bug in the main function", user_info=user_info)
```

### Custom Tool Registration

```python
from cursor_agent_tools import create_agent

agent = create_agent(model='claude-3-5-sonnet-latest')

def custom_tool(param1, param2):
    # Tool implementation
    return {"result": f"Processed {param1} and {param2}"}

# Register the tool
agent.register_tool(
    name="custom_tool",
    function=custom_tool,
    description="Custom tool that does something useful",
    parameters={
        "properties": {
            "param1": {"description": "First parameter", "type": "string"},
            "param2": {"description": "Second parameter", "type": "string"}
        },
        "required": ["param1", "param2"]
    }
)
```

For more examples, check the [examples](examples/) directory.

## 🏗️ Project Structure

```
cursor-agent/
├── agent/                   # Core agent implementation
│   ├── __init__.py          # Package exports
│   ├── base.py              # Base agent class
│   ├── claude_agent.py      # Claude-specific implementation
│   ├── openai_agent.py      # OpenAI-specific implementation
│   ├── factory.py           # Agent factory function
│   ├── permissions.py       # Permission system implementation
│   ├── interact.py          # Interactive mode utilities
│   └── tools/               # Tool implementations
│       ├── __init__.py      # Tool exports
│       ├── file_tools.py    # File operations
│       ├── search_tools.py  # Search functionalities
│       ├── system_tools.py  # System commands
│       └── register_tools.py # Tool registration utilities
├── cursor_agent/            # Package directory for pip installation
│   ├── __init__.py          # Package exports
│   └── agent/               # Re-exports of agent functionality
├── docs/                    # Documentation
│   └── permissions_guide.md # Permission system documentation
├── examples/                # Example usage scripts
│   ├── basic_usage.py       # Simple API usage example
│   ├── chat_conversation_example.py  # Conversation example
│   ├── code_search_example.py  # Code search demonstration
│   ├── file_manipulation_example.py  # File tools example
│   ├── interactive_mode_example.py  # Interactive session demo
│   ├── permission_example.py  # Permission system demonstration
│   ├── simple_task_example.py  # Basic task completion
│   ├── utils.py             # Example utilities
│   └── demo_project/        # Demo project for examples
├── tests/                   # Unit and integration tests
│   ├── test_permissions.py  # Permission system tests
│   └── ...                  # Other test files
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore patterns
├── CODE_OF_CONDUCT.md       # Code of conduct for contributors
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT License
├── README.md                # This file
├── SECURITY.md              # Security policy
├── constraints.md           # Implementation constraints
├── pyproject.toml           # Project configuration
├── requirements.txt         # Project dependencies
├── run_tests.py             # Test runner script
├── run_ci_checks.sh         # CI check script
└── setup.py                 # Package installation
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | None |
| `ANTHROPIC_API_MODEL` | Claude model to use | claude-3-5-sonnet-latest |
| `ANTHROPIC_TEMPERATURE` | Claude temperature setting | 0.0 |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `OPENAI_API_MODEL` | OpenAI model to use | gpt-4o |
| `OPENAI_TEMPERATURE` | OpenAI temperature setting | 0.0 |
| `ENVIRONMENT` | Environment mode | local |

### Agent Configuration

When creating an agent, you can customize its behavior:

```python
from cursor_agent_tools import create_agent
from cursor_agent_tools.permissions import PermissionOptions

# Create permission options
permissions = PermissionOptions(
    yolo_mode=True,
    command_allowlist=["ls", "echo", "git"],
    command_denylist=["rm -rf", "sudo"],
    delete_file_protection=True
)

agent = create_agent(
    model='claude-3-5-sonnet-latest',  # Specific model to use (determines the agent type)
    temperature=0.2,                    # Creativity level
    system_prompt=None,                 # Custom system prompt
    tools=None,                         # Custom tools dictionary
    permission_options=permissions      # Permission configuration
)
```

## 🔐 Permission System

The CursorAgent includes a robust permission system for secure handling of system operations:

### Key Features

- **Secure by Default**: All file modifications and command executions require permission
- **YOLO Mode**: Optional mode for automatic approval of operations (with configurable rules)
- **Command Filtering**: Allowlist/denylist for controlling which commands can run automatically
- **File Deletion Protection**: Special protection for file deletion operations
- **Customizable UI**: Flexible permission request interface adaptable to different environments

### Basic Usage

```python
from cursor_agent_tools import create_agent
from cursor_agent_tools.permissions import PermissionOptions

# Create an agent with default permissions (requires confirmation for all operations)
permissions = PermissionOptions(yolo_mode=False)
agent = create_agent(
    model='claude-3-5-sonnet-latest',
    permission_options=permissions
)

# Create an agent with YOLO mode (many operations auto-approved)
permissions = PermissionOptions(
    yolo_mode=True,
    command_allowlist=["ls", "echo", "git"],
    delete_file_protection=True
)
agent = create_agent(
    model='claude-3-5-sonnet-latest',
    permission_options=permissions
)
```

### Custom Permission Handlers

The permission system can be adapted to different UI environments:

```python
from cursor_agent_tools.permissions import PermissionOptions, PermissionRequest, PermissionStatus

# Create a custom permission handler for a GUI application
def gui_permission_handler(request: PermissionRequest) -> PermissionStatus:
    # Implement GUI-based permission dialog
    # ...
    return PermissionStatus.GRANTED  # or DENIED

# Create permission options with custom handler
permissions = PermissionOptions(
    yolo_mode=False,
    permission_callback=gui_permission_handler
)

agent = create_agent(
    model='claude-3-5-sonnet-latest',
    permission_options=permissions
)
```

For comprehensive documentation on the permission system, see [permissions_guide.md](docs/permissions_guide.md).

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

### Development Setup

1. Fork and clone the repository
2. Create a virtual environment
3. Install development dependencies: `pip install -e ".[dev]"`
4. Set up API keys in a `.env` file
5. Submit pull requests for features or fixes

### Code Style

We use flake8 for linting. To ensure consistent style, please make sure your code:

1. Passes the flake8 linting check with `flake8 cursor_agent_tools`
2. Has no whitespace issues. You can use the included script to fix common whitespace issues:
   ```
   python fix_whitespace_errors.py
   ```
   This will automatically fix trailing whitespace (W291) and blank lines with whitespace (W293) in the cursor_agent_tools directory.

## 📚 API Documentation

### Agent Factory

```python
# Import the factory function
from cursor_agent_tools import create_agent

def create_agent(
    model: str,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[Dict] = None,
) -> BaseAgent:
    """
    Create an agent instance based on the specified model.
    
    Args:
        model: The model to use (e.g. 'claude-3-5-sonnet-latest', 'gpt-4o')
        temperature: The temperature setting for generation
        system_prompt: Custom system prompt to use
        tools: Custom tools dictionary
        
    Returns:
        An instance of BaseAgent (either ClaudeAgent or OpenAIAgent)
    """
```

### Base Agent Methods

```python
# Import necessary types
from typing import Dict, List, Callable, Optional
from cursor_agent_tools import BaseAgent

class BaseAgent:
    async def chat(
        self, user_message: str, user_info: Optional[Dict] = None
    ) -> str:
        """Send a message to the agent and get a response."""
        
    def register_tool(
        self, name: str, function: Callable, description: str, parameters: Dict
    ) -> None:
        """Register a custom tool with the agent."""
        
    async def _prepare_tools(self) -> Dict:
        """Prepare tools for the model's API format."""
        
    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
```

For complete API documentation, refer to the docstrings in the source code.

## 🔧 Advanced Usage

### Tool Implementation

Tools are Python functions registered with the agent. The cursor-agent library supports various types of tools to enhance your agent's capabilities:

#### Basic Tool Registration

```python
from cursor_agent_tools import create_agent

agent = create_agent(model='claude-3-5-sonnet-latest')

def database_query(query: str, connection_string: str):
    """Execute a database query and return results."""
    # Implementation...
    return {"results": [...]}

agent.register_tool(
    name="database_query",
    function=database_query,
    description="Execute a SQL query against a database",
    parameters={
        "properties": {
            "query": {"description": "SQL query to execute", "type": "string"},
            "connection_string": {"description": "Database connection string", "type": "string"}
        },
        "required": ["query", "connection_string"]
    }
)
```

#### API Integration Tools

```python
from cursor_agent_tools import create_agent
import requests

agent = create_agent(model='claude-3-5-sonnet-latest')

def fetch_weather(location: str, units: str = "metric"):
    """Fetch current weather data for a location."""
    API_KEY = "your_api_key"  # Better to use environment variables
    url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={location}&units={units}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API returned status code {response.status_code}"}

agent.register_tool(
    name="fetch_weather",
    function=fetch_weather,
    description="Get current weather data for a specified location",
    parameters={
        "properties": {
            "location": {"description": "City name or coordinates", "type": "string"},
            "units": {"description": "Units system (metric or imperial)", "type": "string"}
        },
        "required": ["location"]
    }
)
```

#### Data Processing Tools

```python
from cursor_agent_tools import create_agent
import pandas as pd
import json

agent = create_agent(model='claude-3-5-sonnet-latest')

def analyze_csv(file_path: str, operations: list):
    """Perform analytical operations on a CSV file."""
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        results = {}
        for operation in operations:
            if operation == "summary":
                results["summary"] = json.loads(df.describe().to_json())
            elif operation == "columns":
                results["columns"] = df.columns.tolist()
            elif operation == "missing":
                results["missing"] = json.loads(df.isnull().sum().to_json())
                
        return results
    except Exception as e:
        return {"error": str(e)}

agent.register_tool(
    name="analyze_csv",
    function=analyze_csv,
    description="Analyze a CSV file with various statistical operations",
    parameters={
        "properties": {
            "file_path": {"description": "Path to the CSV file", "type": "string"},
            "operations": {
                "description": "List of operations to perform",
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["file_path", "operations"]
    }
)
```

### File Operations Examples

#### Line-Based File Editing

The agent supports precise editing of files using line numbers:

```python
import json
from cursor_agent_tools import create_agent

async def main():
    agent = create_agent(model='claude-3-5-sonnet-latest')
    
    # Define line-based edits as a dictionary with line ranges as keys
    line_edits = {
        "5-8": "def calculate_total(items):\n    \"\"\"Calculate the total price of all items.\"\"\"\n    return sum(item.price for item in items)\n",
        "12-12": "    # Log the transaction\n    logging.info(f\"Processed order: {order_id}\")\n"
    }
    
    # Convert to JSON string for the edit_file function
    code_edit_json = json.dumps(line_edits)
    
    # Apply edits to specific line ranges
    await agent.edit_file(
        target_file="/path/to/your/file.py",
        instructions="Update calculate_total function and add logging",
        code_edit=code_edit_json
    )

if __name__ == "__main__":
    asyncio.run(main())
```

This approach has several advantages:
- Precisely target specific line ranges for editing
- Make multiple edits in a single operation
- Clear and structured format for programmatic editing
- Easier to automate and script file modifications

For more examples, check out [line_based_edit_example.py](examples/line_based_edit_example.py).

### Providing Project Context

```python
from cursor_agent_tools import create_agent

agent = create_agent(model='claude-3-5-sonnet-latest')

user_info = {
    "open_files": ["src/main.py", "src/utils.py"],
    "cursor_position": {"file": "src/main.py", "line": 42},
    "recent_files": ["src/config.py", "tests/test_main.py"],
    "os": "darwin",
    "workspace_path": "/Users/username/projects/myproject"
}

response = await agent.chat("Fix the bug in the main function", user_info=user_info)
```

## ⚠️ Limitations and Considerations

- **API Key Security**: Keep your API keys secure and never commit them to repositories
- **Context Windows**: Models have token limits that restrict the amount of code they can process
- **Tool Execution**: Function calling executes code on your system - implement proper security
- **Rate Limits**: APIs have rate limits that may restrict usage
- **Costs**: Using the APIs incurs costs based on token usage

For a detailed list of constraints and workarounds, see the [constraints.md](constraints.md) file.

## 🛣️ Roadmap

- Streaming responses
- Support for more model families (e.g., Gemini, Llama, etc.)
- Web interface
- Multi-user support with authentication
- Vector-based codebase search
- Testing tools integration
- Comprehensive demo suite
- Documentation website

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Anthropic](https://www.anthropic.com/) for the Claude API
- [OpenAI](https://openai.com/) for the OpenAI API
- [Cursor](https://cursor.sh/) for inspiration

## 👤 Author

Femi Amoo (Nifemi Alpine)

Founder of [CIVAI TECHNOLOGIES](https://civai.co)

[![Twitter](https://img.shields.io/twitter/follow/usecodenaija?style=social)](https://twitter.com/usecodenaija) 

### Use You.com Search API with `web_search`

Set the provider and optional API key:

```bash
export WEB_SEARCH_PROVIDER=you
# Optional: up to 100 free searches/day without key
export YDC_API_KEY=your_key_here
```

The `web_search` tool will then query `https://api.you.com/v1/agents/search` and return normalized results in the same response shape used by existing integrations.
