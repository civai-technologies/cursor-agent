from setuptools import setup, find_packages
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Replace relative links with absolute GitHub URLs
repo_url = "https://github.com/civai-technologies/cursor-agent"
branch = "main"  # Assuming main is the default branch

# Process in this order:
# 1. First handle image links separately (which use ![ syntax)
img_pattern = r'!\[([^\]]+)\]\((?!https?://)([^)]+)\)'
img_replacement = lambda m: f'![{m.group(1)}]({repo_url}/raw/{branch}/{m.group(2)})'
long_description = re.sub(img_pattern, img_replacement, long_description)

# 2. Replace directory links (paths ending with /)
dir_with_slash_pattern = r'\[([^\]]+)\]\((?!https?://|#)([^)]+/)\)'
dir_with_slash_replacement = lambda m: f'[{m.group(1)}]({repo_url}/tree/{branch}/{m.group(2)})'
long_description = re.sub(dir_with_slash_pattern, dir_with_slash_replacement, long_description)

# 3. Replace directory links without trailing slash - looking for directories referenced in the TOC
# This targets paths without extensions that are likely directories
dir_pattern = r'\[([^\]]+)\]\((?!https?://|#)([^.)]+)\)'
dir_replacement = lambda m: f'[{m.group(1)}]({repo_url}/tree/{branch}/{m.group(2)})'
long_description = re.sub(dir_pattern, dir_replacement, long_description)

# 4. Finally replace remaining file links
file_pattern = r'\[([^\]]+)\]\((?!https?://|#)([^)]+\.[a-zA-Z0-9]+[^)]*)\)'
file_replacement = lambda m: f'[{m.group(1)}]({repo_url}/blob/{branch}/{m.group(2)})'
long_description = re.sub(file_pattern, file_replacement, long_description)

setup(
    name="cursor-agent",
    version="1.0.0",
    author="Nifemi Alpine",
    author_email="hello@civai.co",
    description="A Python-based AI agent that replicates Cursor's coding assistant capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/civai-technologies/cursor-agent",
    packages=["cursor_agent", "agent", "agent.tools"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.49.0",
        "openai>=1.6.1",
        "colorama>=0.4.6",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.8.0",
        "requests>=2.31.0",
        "urllib3>=2.0.7",
        "httpx>=0.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "bump2version>=1.0.0",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
