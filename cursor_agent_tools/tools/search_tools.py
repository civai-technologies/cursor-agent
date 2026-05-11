import json
import os
import re
import subprocess
import requests
from typing import Any, Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

from ..logger import get_logger

# Define exported functions
__all__ = [
    "codebase_search",
    "grep_search",
    "file_search",
    "web_search",
    "trend_search",
    "get_trending_topics"
]

# Initialize logger
logger = get_logger(__name__)


def codebase_search(
    query: str, target_directories: Optional[List[str]] = None, explanation: Optional[str] = None, agent: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Find snippets of code from the codebase most relevant to the search query.
    This is a semantic search tool that finds code semantically matching the query.

    Args:
        query: The search query to find relevant code
        target_directories: Optional list of directories to search in
        explanation: Optional explanation of why this search is being performed
        agent: Reference to the agent instance (unused in this function but kept for consistency)

    Returns:
        Dict containing the search results
    """
    try:
        logger.info(f"Performing codebase search for: {query}")

        if target_directories is None:
            # Default to current directory if none specified
            target_directories = [os.getcwd()]
            logger.debug(f"No target directories specified, using current directory: {os.getcwd()}")
        else:
            logger.debug(f"Searching in directories: {', '.join(target_directories)}")

        # For now, we'll use a simple grep-based approach since we don't have a semantic search engine
        # In a real implementation, this should use a vector search or dedicated code search tool
        results = []

        for directory in target_directories:
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
                continue

            for root, _, files in os.walk(directory):
                for file in files:
                    # Skip binary files and hidden files
                    if file.startswith(".") or any(
                        file.endswith(ext) for ext in [".jpg", ".png", ".gif", ".zip", ".pyc"]
                    ):
                        continue

                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Very simple search - in a real implementation, use semantic search
                        if query.lower() in content.lower():
                            # Find the line numbers where the query appears
                            lines = content.splitlines()
                            matches = []

                            for i, line in enumerate(lines):
                                if query.lower() in line.lower():
                                    context_start = max(0, i - 2)
                                    context_end = min(len(lines) - 1, i + 2)
                                    context = "\n".join(lines[context_start : context_end + 1])
                                    matches.append(
                                        {
                                            "line_number": i + 1,  # 1-indexed
                                            "content": line,
                                            "context": context,
                                        }
                                    )

                            if matches:
                                results.append(
                                    {
                                        "file": file_path,
                                        "matches": matches[:5],  # Limit to 5 matches per file
                                    }
                                )
                                logger.debug(f"Found {len(matches)} matches in file: {file_path}")
                    except Exception as e:
                        # Skip files that can't be read
                        logger.debug(f"Error reading file {file_path}: {str(e)}")
                        continue

        logger.info(f"Codebase search completed. Found relevant code in {len(results)} files")
        return {
            "query": query,
            "results": results[:20],  # Limit to 20 files
            "total_files_searched": sum(
                1 for _ in os.walk(directory) for directory in target_directories
            ),
        }

    except Exception as error:
        logger.error(f"Error in codebase search: {str(error)}")
        return {"error": str(error)}


def grep_search(
    query: str,
    explanation: Optional[str] = None,
    case_sensitive: bool = False,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    agent: Optional[Any] = None  # Optional agent reference
) -> Dict[str, Any]:
    """
    Fast text-based regex search that finds exact pattern matches within files or directories.

    Args:
        query: The regex pattern to search for
        explanation: Optional explanation of why this search is being performed
        case_sensitive: Whether the search should be case sensitive
        include_pattern: Optional glob pattern for files to include
        exclude_pattern: Optional glob pattern for files to exclude
        agent: Reference to the agent instance (unused in this function but kept for consistency)

    Returns:
        Dict containing the search results
    """
    try:
        logger.info(f"Performing grep search for pattern: {query}")
        logger.debug(f"Search parameters - case_sensitive: {case_sensitive}, include: {include_pattern}, exclude: {exclude_pattern}")

        # Check if ripgrep is installed
        have_ripgrep = False
        try:
            subprocess.run(["rg", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            have_ripgrep = True
            logger.debug("Using ripgrep for search")
        except (subprocess.SubprocessError, FileNotFoundError):
            have_ripgrep = False
            logger.debug("Ripgrep not available, using fallback search")

        results = []

        if have_ripgrep:
            # Use ripgrep for faster searching
            cmd = ["rg", "--json"]

            if not case_sensitive:
                cmd.append("-i")

            if include_pattern:
                cmd.extend(["-g", include_pattern])

            if exclude_pattern:
                cmd.extend(["-g", f"!{exclude_pattern}"])

            cmd.extend(["--max-count", "50", query, "."])

            logger.debug(f"Executing ripgrep command: {' '.join(cmd)}")
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = process.stdout

            # Parse the JSON output
            for line in output.splitlines():
                try:
                    data = json.loads(line)
                    if data["type"] == "match":
                        file_path = data["data"]["path"]["text"]
                        line_number = data["data"]["line_number"]
                        line_text = data["data"]["lines"]["text"]

                        results.append(
                            {
                                "file": file_path,
                                "line_number": line_number,
                                "content": line_text.strip(),
                            }
                        )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug(f"Error parsing ripgrep output: {str(e)}")
                    continue
        else:
            # Fallback to a simple recursive grep
            for root, _, files in os.walk(os.getcwd()):
                for file in files:
                    # Apply include/exclude filters
                    if include_pattern and not re.match(include_pattern, file):
                        continue

                    if exclude_pattern and re.match(exclude_pattern, file):
                        continue

                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()

                        for i, line in enumerate(lines):
                            flags = 0 if case_sensitive else re.IGNORECASE
                            if re.search(query, line, flags=flags):
                                results.append(
                                    {
                                        "file": file_path,
                                        "line_number": i + 1,  # 1-indexed
                                        "content": line.strip(),
                                    }
                                )
                    except Exception as e:
                        # Skip files that can't be read
                        logger.debug(f"Error reading file {file_path}: {str(e)}")
                        continue

                    # Limit to 50 matches
                    if len(results) >= 50:
                        break

                if len(results) >= 50:
                    break

        logger.info(f"Grep search completed. Found {len(results)} matches")
        return {"query": query, "results": results, "total_matches": len(results)}

    except Exception as error:
        logger.error(f"Error in grep search: {str(error)}")
        return {"error": str(error)}


def file_search(
    query: str,
    explanation: Optional[str] = None,
    agent: Optional[Any] = None  # Optional agent reference
) -> Dict[str, Any]:
    """
    Fast file search based on fuzzy matching against file path.

    Args:
        query: Fuzzy filename to search for
        explanation: Optional explanation of why this search is being performed
        agent: Reference to the agent instance (unused in this function but kept for consistency)

    Returns:
        Dict containing the search results
    """
    try:
        logger.info(f"Performing file search for: {query}")

        results = []

        for root, _, files in os.walk(os.getcwd()):
            for file in files:
                if query.lower() in file.lower():
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    file_type = "unknown"

                    # Determine file type based on extension
                    if "." in file:
                        extension = file.split(".")[-1].lower()
                        file_type = extension

                    results.append(
                        {"path": file_path, "name": file, "size": file_size, "type": file_type}
                    )
                    logger.debug(f"Found matching file: {file_path}")

                    # Limit to 10 results
                    if len(results) >= 10:
                        break

            if len(results) >= 10:
                break

        logger.info(f"File search completed. Found {len(results)} matching files")
        return {"query": query, "results": results, "total_matches": len(results)}

    except Exception as error:
        logger.error(f"Error in file search: {str(error)}")
        return {"error": str(error)}


def web_search(
    search_term: str,
    explanation: Optional[str] = None,
    force: bool = False,
    objective: Optional[str] = None,
    max_results: int = 5,
    agent: Optional[Any] = None  # Optional agent reference
) -> Dict[str, Any]:
    """
    Search the web for up-to-date information about any topic using Google Custom Search API.

    Args:
        search_term: The search term to look up on the web
        explanation: Optional explanation of why this search is being performed
        force: Force internet access even if not required
        objective: Optional user objective to determine if up-to-date data is needed
        max_results: Maximum number of results to return
        agent: Reference to the agent instance for LLM queries

    Returns:
        Dict containing the search results with content from top websites
    """
    try:
        logger.info(f"Performing web search for: {search_term}")

        # Provider selection: default to Google for backward compatibility
        search_provider = os.environ.get("WEB_SEARCH_PROVIDER", "google").strip().lower()
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        google_search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
        ydc_api_key = os.environ.get("YDC_API_KEY")

        if search_provider not in ["google", "you"]:
            return {"error": "Unsupported WEB_SEARCH_PROVIDER. Use 'google' or 'you'.", "results": []}

        if search_provider == "google" and (not google_api_key or not google_search_engine_id):
            logger.error("Missing Google API key or Search Engine ID in environment variables")
            return {
                "error": "Missing API keys. Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables.",
                "results": []
            }

        # Check if this search requires up-to-date information
        requires_up_to_date = force

        if objective and not force and agent:
            # Since we're not using the agent.chat anymore, we don't need this prompt
            # up_to_date_prompt = f"Based on the main goal: {objective}\nDoes the following research objective require the most recent information?\n\n'{search_term}' \n\nAnswer in a single word, 'yes' or 'no'."

            # Just assume most queries need up-to-date info for simplicity
            requires_up_to_date = True
            logger.debug(f"Assuming up-to-date information is needed: {requires_up_to_date}")

        if not requires_up_to_date and not force:
            logger.info("Search does not require up-to-date information, skipping web search")
            return {
                "results": [],
                "message": "This query doesn't require up-to-date information. Set force=True to force a web search."
            }

        if search_provider == "you":
            logger.info("Performing internet search using You.com Search API...")
            search_results = you_search_sync(search_term, ydc_api_key, max_results=max_results)
        else:
            logger.info("Performing internet search using Google Custom Search API...")
            search_results = google_search_sync(search_term, google_api_key, google_search_engine_id, max_results=max_results)

        if not search_results:
            logger.warning("No search results found")
            return {
                "results": [],
                "message": "No search results found"
            }

        # Scrape content from search results
        content_summaries = scrape_content_sync(search_results)

        # Format results
        results = []
        for url, summary in content_summaries.items():
            if url in search_results:
                title = search_results[url].get('title', 'Unknown Title')
                results.append({
                    "title": title,
                    "url": url,
                    "content": summary
                })

        logger.info(f"Web search completed. Found {len(results)} relevant results")
        return {
            "query": search_term,
            "results": results,
            "total_results": len(results)
        }

    except Exception as error:
        logger.error(f"Error in web search: {str(error)}")
        return {"error": str(error), "results": []}


def google_search_sync(query: str, api_key: str, search_engine_id: str, max_results: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Perform a search using Google Custom Search API synchronously with pagination support.

    Args:
        query: The search query
        api_key: Google API key
        search_engine_id: Google Search Engine ID
        max_results: Maximum number of results to return (will paginate if > 10)

    Returns:
        Dictionary mapping URLs to search result data
    """
    try:
        logger.info(f"Performing Google Custom Search for: {query} (max_results: {max_results})")

        all_results = {}
        results_collected = 0
        start_index = 1  # Google API uses 1-based indexing

        # Calculate how many API calls we need (max 10 results per call)
        while results_collected < max_results:
            # Calculate how many results to request in this call
            results_needed = max_results - results_collected
            results_per_call = min(10, results_needed)  # API limit is 10 per call

            logger.info(f"Making API call - start: {start_index}, num: {results_per_call}")

            url = "https://www.googleapis.com/customsearch/v1"
            params: Dict[str, str] = {
                'key': api_key,
                'cx': search_engine_id,
                'q': query,
                'num': str(results_per_call),
                'start': str(start_index)
            }

            response = requests.get(url, params=params)
            logger.info(f"API Response Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])

                if not items:
                    logger.info(f"No more results available. Collected {results_collected} results total.")
                    break

                logger.info(f"Retrieved {len(items)} results in this call")

                # Process results from this call
                for item in items:
                    link = item.get('link', '')
                    title = item.get('title', '')
                    snippet = item.get('snippet', '')

                    if link:
                        all_results[link] = {
                            'title': title,
                            'snippet': snippet,
                            'url': link
                        }
                        results_collected += 1

                        # Stop if we've collected enough results
                        if results_collected >= max_results:
                            break

                # Prepare for next API call
                start_index += len(items)

                # Check if we've reached the end of available results
                total_results = data.get('searchInformation', {}).get('totalResults', '0')
                if start_index > int(total_results):
                    logger.info(f"Reached end of available results. Total available: {total_results}")
                    break

            else:
                logger.error(f"Google Search API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                break

        logger.info(f"Google Search completed. Total results collected: {len(all_results)}")
        return all_results

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during Google search: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error during Google search: {e}")
        return {}


def you_search_sync(query: str, api_key: Optional[str], max_results: int = 5) -> Dict[str, Dict[str, Any]]:
    """Perform a search using You.com Search API synchronously."""
    try:
        logger.info(f"Performing You.com Search for: {query} (max_results: {max_results})")
        params = {"query": query, "count": str(max_results)}
        headers: Dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key

        response = requests.get("https://api.you.com/v1/agents/search", params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            logger.error(f"You.com Search API error {response.status_code}: {response.text}")
            return {}

        data = response.json()
        results: Dict[str, Dict[str, Any]] = {}
        for section in ["web", "news"]:
            for item in data.get("results", {}).get(section, [])[:max_results]:
                url = item.get("url")
                if not url:
                    continue
                results[url] = {
                    "title": item.get("title", "Unknown Title"),
                    "snippet": item.get("description", ""),
                }
        return results
    except Exception as e:
        logger.error(f"Unexpected error during You.com search: {e}")
        return {}


def scrape_content_sync(search_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Scrape and summarize content from search result URLs synchronously.

    Args:
        search_results: Dictionary mapping URLs to search result data

    Returns:
        Dictionary mapping URLs to content summaries
    """
    content_summaries = {}

    for url in search_results:
        try:
            logger.info(f"Scraping content from: {url}")

            # Request the page with a timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: {response.status_code}")
                continue

            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get the text
            text = soup.get_text()

            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Limit text to a reasonable size (first 5000 chars)
            text = text[:5000]

            # Store the result
            content_summaries[url] = text

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")

    return content_summaries


async def trend_search(
    query: str,
    explanation: Optional[str] = None,
    country_code: str = "US",
    days: int = 7,
    max_results: int = 3,
    lookback_hours: int = 48,
    agent: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Search for trending topics related to the query by:
    1. Determining the trend category
    2. Getting a list of trends in that category
    3. Searching for content about each individual trend

    Args:
        query: The user's query about trends
        explanation: Optional explanation of why this search is being performed
        country_code: Country code for trends (default: US)
        days: Number of days to look back for trends (default: 7)
        max_results: Maximum number of trends to return (default: 3)
        lookback_hours: Number of hours to look back for Google Trends data (default: 48)
        agent: Reference to the agent instance for categorizing the query

    Returns:
        Dict containing trend information with content from relevant sources
    """
    try:
        logger.info(f"🔍 Starting trend search for query: '{query}'")

        # List of available categories with their IDs
        categories = {
            "All Categories": None,
            "Science": 15,
            "Technology": 18,
            "Jobs & Education": 9,
            "Entertainment": 4,
            "Food & Drink": 5,
            "Business & Finance": 3,
            "Beauty & Fashion": 2,
            "Sports": 17,
            "Autos & Vehicles": 1,
            "Climate": 20,
            "Games": 6,
            "Health": 7,
            "Hobbies & Leisure": 8,
            "Law & Government": 10,
            "Other": 11,
            "Pets & Animals": 13,
            "Politics": 14,
            "Shopping": 16,
            "Travel & Transportation": 19
        }

        # Step 1: Determine the trend category using the agent
        if not agent:
            logger.warning("❗ No agent provided for trend categorization, defaulting to Arts & Entertainment")
            category_name = "Arts & Entertainment"
            category_id = 4
        else:
            category_name, category_id = await _determine_trend_category(query, categories, agent)
            logger.info(f"🏷️ Determined category: {category_name} (ID: {category_id})")

        # Step 2: Get a list of trends in this category
        # For now, we'll use web search to get potential trends since we don't have direct API access
        search_term = f"top trending topics in {category_name} {days} days"
        if country_code and country_code != "US":
            search_term += f" in {country_code}"

        logger.info(f"📈 Getting trending topics for: '{search_term}'")

        # Get potential trends in this category
        trends = await get_trending_topics(search_term, category_name, country_code, lookback_hours, agent)

        if not trends:
            logger.warning(f"❌ No trends found in category {category_name}")
            return {
                "query": query,
                "category": category_name,
                "category_id": category_id,
                "country_code": country_code,
                "days": days,
                "trends": [],
                "total_trends": 0
            }

        logger.info(f"🎯 Found {len(trends)} potential trends in {category_name}")

        # Step 3: Search for content about each trend
        processed_trends = []

        for i, trend in enumerate(trends[:max_results], 1):
            logger.info(f"\n👉 TREND #{i}: '{trend}'")
            logger.info("   🔎 Searching for information about this trend...")

            # Search for information about the specific trend
            search_results = web_search(
                search_term=trend,
                explanation="Finding information about {0}".format(trend),
                force=True,
                max_results=max_results
            )

            # Check for API key error in search results
            if "error" in search_results and "Missing API keys" in search_results["error"]:
                error_msg = "Google API keys are required for trend search. Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables."
                logger.error(error_msg)
                return {
                    "query": query,
                    "category": category_name,
                    "category_id": category_id,
                    "country_code": country_code,
                    "days": days,
                    "trends": [],
                    "total_trends": 0,
                    "error": error_msg
                }

            if not search_results or not search_results.get("results"):
                logger.warning(f"   ❌ No search results found for trend: '{trend}'. Skipping.")
                continue

            # Get first result as the main snippet
            first_result = search_results["results"][0]
            snippet = first_result.get("content", "")[:300]  # Limit snippet size

            logger.info(f"   📝 Got main snippet: '{snippet[:100]}...'")
            logger.info(f"   🌐 Found {len(search_results.get('results', []))} sources")

            # Create a dictionary mapping URLs to their content
            content_data = {}
            for result in search_results["results"]:
                url = result.get("url", "")
                content = result.get("content", "")
                if url and content:
                    content_data[url] = content
                    logger.debug(f"   Source: {url} ({len(content)} chars)")

            # Create trend data
            trend_data: Dict[str, Any] = {
                "name": trend,
                "snippet": snippet,
                "sources": list(content_data.keys()),
                "content": content_data
            }

            processed_trends.append(trend_data)
            logger.info(f"   ✅ Successfully processed trend: '{trend}'")

        result_count = len(processed_trends)
        logger.info(f"🏁 Completed trend search. Processed {result_count} trends with content")

        if result_count > 0:
            logger.info("📊 SUMMARY OF PROCESSED TRENDS:")
            for i, trend_data in enumerate(processed_trends, 1):
                # Ensure we.re getting from a Dict, not a string
                if isinstance(trend_data, dict):
                    name = trend_data.get('name', '')
                    snippet = trend_data.get('snippet', '')
                    sources = trend_data.get('sources', [])
                    logger.info(f"   #{i}: '{name}'")
                    logger.info(f"      Snippet: '{snippet[:100]}...'")
                    logger.info(f"      Sources: {len(sources)}")

        return {
            "query": query,
            "category": category_name,
            "category_id": category_id,
            "country_code": country_code,
            "days": days,
            "trends": processed_trends,
            "total_trends": len(processed_trends)
        }

    except Exception as error:
        logger.error(f"❌ Error in trend search: {str(error)}")
        return {
            "query": query,
            "category": "Entertainment",  # Default category
            "category_id": 4,
            "country_code": country_code,
            "days": days,
            "trends": [],
            "total_trends": 0,
            "error": str(error)
        }


async def get_trending_topics(search_term: str, category: str, country_code: str = 'US', lookback_hours: int = 48, agent: Optional[Any] = None) -> List[str]:
    """
    Get a list of trending topics in a category using Google Trends API.

    Args:
        search_term: Search term (only used for logging)
        category: Category name
        country_code: Country code for localized trends
        lookback_hours: Number of hours to look back for trends (default: 48)
        agent: Optional agent for extraction

    Returns:
        List of trending topic names
    """
    try:
        # Map category names to Google Trends category IDs
        category_id_map = {
            "All Categories": None,
            "Science": 15,
            "Technology": 18,
            "Jobs & Education": 9,
            "Entertainment": 4,
            "Food & Drink": 5,
            "Business & Finance": 3,
            "Beauty & Fashion": 2,
            "Sports": 17,
            "Autos & Vehicles": 1,
            "Climate": 20,
            "Games": 6,
            "Health": 7,
            "Hobbies & Leisure": 8,
            "Law & Government": 10,
            "Other": 11,
            "Pets & Animals": 13,
            "Politics": 14,
            "Shopping": 16,
            "Travel & Transportation": 19
        }

        # Get the corresponding Google Trends category ID
        trends_category_id = category_id_map.get(category, 4)  # Default to Entertainment

        logger.info(f"📊 Fetching trending topics in {category} (ID: {trends_category_id}) for {country_code}")

        # Import needed libraries
        import requests

        # Use the TrendsUi batchexecute endpoint with source-path for better results
        url = "https://trends.google.com/_/TrendsUi/data/batchexecute?source-path=/trending"

        # Setting up payload with the selected category ID
        payload = f'f.req=[[["i0OFE","[null, null, \\"{country_code}\\", {trends_category_id}, \\"en-US\\", {lookback_hours}, 1]"]]]'

        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        logger.info("🔍 Connecting to Google Trends API for category {0}...".format(trends_category_id))

        # Make the request
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()

        # Extract JSON from response
        trends_data = _extract_json_from_trends_response(response.text)

        if not trends_data:
            logger.warning("❌ Failed to extract trends data from response")
            return []

        # Extract search terms with their search volume and category
        trends_with_volume: List[Dict[str, Any]] = []
        for item in trends_data:
            try:
                if len(item) < 7:  # Needs at least 7 elements to have search volume
                    continue

                search_term = item[0].lower()
                search_volume = item[6] if isinstance(item[6], int) else 0

                # Category information is in element 10
                category_ids = []
                if len(item) > 10 and isinstance(item[10], list):
                    category_ids = item[10]

                # Add to our list
                if trends_category_id in category_ids:
                    trends_with_volume.append({
                        "term": search_term,
                        "volume": search_volume,
                        "category_ids": category_ids
                    })

            except Exception as e:
                logger.warning(f"⚠️ Error processing trend item: {str(e)}")
                continue

        # Sort by search volume (highest first)
        sorted_trends = sorted(trends_with_volume, key=lambda x: x.get("volume", 0), reverse=True)

        # Get the top 10 search terms
        top_terms: List[str] = [str(item["term"]) for item in sorted_trends[:10]]

        # Log results
        logger.info(f"✅ Retrieved {len(top_terms)} trending searches sorted by volume")
        for i, trend in enumerate(top_terms[:5], 1):
            trend_info = next((t for t in sorted_trends if t["term"] == trend), None)
            if trend_info:
                logger.info(f"   #{i}: {trend} (volume: {trend_info['volume']})")
            else:
                logger.info(f"   #{i}: {trend}")

        if len(top_terms) > 5:
            logger.info(f"   ... and {len(top_terms) - 5} more")

        return top_terms

    except Exception as e:
        logger.error(f"❌ Error fetching trends using Google Trends API: {str(e)}")

        # If Google Trends API fails, fall back to using the agent or web search
        logger.info("⚠️ Falling back to alternative trend discovery method")
        return ["Error fetching trends using Google Trends API: {0}".format(str(e))]


def _extract_json_from_trends_response(text: str) -> List[Any]:
    """
    Extracts the nested JSON object from the Google Trends API response.

    Args:
        text: The response text from the API

    Returns:
        Parsed JSON data containing trends
    """
    for line in text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith('[') and trimmed.endswith(']'):
            try:
                # First level JSON parsing
                intermediate = json.loads(trimmed)
                # Extract and parse the nested JSON string
                data = json.loads(intermediate[0][2])
                # The trends data is in the second element
                # Explicitly type as List[Any]
                result: List[Any] = data[1]
                return result
            except Exception as e:
                logger.warning(f"⚠️ Error parsing JSON from Trends response: {str(e)}")
                continue

    # Return empty list if nothing found
    return []


async def _determine_trend_category(query: str, categories: Dict[str, Optional[int]], agent: Any) -> Tuple[str, int]:
    """
    Use the agent to determine the most appropriate category for the query.

    Args:
        query: The user's query
        categories: Dictionary mapping category names to IDs
        agent: The agent instance

    Returns:
        Tuple of (category_name, category_id)
    """
    try:
        # Define schema for category selection
        category_schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": list(categories.keys()),
                    "description": "The selected category that best matches the query"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this category was selected"
                }
            },
            "required": ["category"]
        }

        # Create prompt for category determination
        prompt = f"""
        Analyze the following query about trends and select the most appropriate category from the available options:

        Query: {query}

        Choose the category that best represents the kind of trends the user is looking for.
        """

        # Get structured output with schema validation
        result = await agent.get_structured_output(prompt, category_schema)

        if result and "category" in result:
            category = result["category"]
            category_id = categories.get(category, 4)  # Default to Arts & Entertainment (4) if not found

            # Ensure we return an int for category_id (not None)
            return category, 4 if category_id is None else category_id

        # Fallback to Arts & Entertainment if no valid response
        logger.warning("Could not determine category, using default")
        return "Arts & Entertainment", 4

    except Exception as e:
        logger.error(f"Error determining category: {str(e)}")
        return "Arts & Entertainment", 4
