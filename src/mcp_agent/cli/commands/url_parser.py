"""
URL parsing utility for the fast-agent CLI.
Provides functions to parse URLs and determine MCP server configurations.
"""

import hashlib
import re
from typing import Dict, List, Literal, Tuple
from urllib.parse import urlparse

from mcp_agent.mcp.hf_auth import add_hf_auth_header


def parse_server_url(
    url: str,
) -> Tuple[str, Literal["http", "sse"], str]:
    """
    Parse a server URL and determine the transport type and server name.

    Args:
        url: The URL to parse

    Returns:
        Tuple containing:
        - server_name: A generated name for the server
        - transport_type: Either "http" or "sse" based on URL
        - url: The parsed and validated URL

    Raises:
        ValueError: If the URL is invalid or unsupported
    """
    # Basic URL validation
    if not url:
        raise ValueError("URL cannot be empty")

    # Parse the URL
    parsed_url = urlparse(url)

    # Ensure scheme is present and is either http or https
    if not parsed_url.scheme or parsed_url.scheme not in ("http", "https"):
        raise ValueError(f"URL must have http or https scheme: {url}")

    # Ensure netloc (hostname) is present
    if not parsed_url.netloc:
        raise ValueError(f"URL must include a hostname: {url}")

    # Determine transport type based on URL path
    transport_type: Literal["http", "sse"] = "http"
    if parsed_url.path.endswith("/sse"):
        transport_type = "sse"
    elif not parsed_url.path.endswith("/mcp"):
        # If path doesn't end with /mcp or /sse, append /mcp
        url = url if url.endswith("/") else f"{url}/"
        url = f"{url}mcp"

    # Generate a server name based on hostname and port
    server_name = generate_server_name(url)

    return server_name, transport_type, url


def generate_server_name(url: str) -> str:
    """
    Generate a unique and readable server name from a URL.

    Args:
        url: The URL to generate a name for

    Returns:
        A server name string
    """
    parsed_url = urlparse(url)

    # Extract hostname and port
    hostname = parsed_url.netloc.split(":")[0]

    # Clean the hostname for use in a server name
    # Replace non-alphanumeric characters with underscores
    clean_hostname = re.sub(r"[^a-zA-Z0-9]", "_", hostname)

    if len(clean_hostname) > 15:
        clean_hostname = clean_hostname[:9] + clean_hostname[-5:]

    # If it's localhost or an IP, add a more unique identifier
    if clean_hostname in ("localhost", "127_0_0_1") or re.match(r"^(\d+_){3}\d+$", clean_hostname):
        # Use the path as part of the name for uniqueness
        path = parsed_url.path.strip("/")
        path = re.sub(r"[^a-zA-Z0-9]", "_", path)

        # Include port if specified
        port = ""
        if ":" in parsed_url.netloc:
            port = f"_{parsed_url.netloc.split(':')[1]}"

        if path:
            return f"{clean_hostname}{port}_{path[:20]}"  # Limit path length
        else:
            # Use a hash if no path for uniqueness
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            return f"{clean_hostname}{port}_{url_hash}"

    return clean_hostname


def parse_server_urls(
    urls_param: str, auth_token: str = None
) -> List[Tuple[str, Literal["http", "sse"], str, Dict[str, str] | None]]:
    """
    Parse a comma-separated list of URLs into server configurations.

    Args:
        urls_param: Comma-separated list of URLs
        auth_token: Optional bearer token for authorization

    Returns:
        List of tuples containing (server_name, transport_type, url, headers)

    Raises:
        ValueError: If any URL is invalid
    """
    if not urls_param:
        return []

    # Split by comma and strip whitespace
    url_list = [url.strip() for url in urls_param.split(",")]

    # Prepare headers if auth token is provided
    headers = None
    if auth_token:
        headers = {"Authorization": f"Bearer {auth_token}"}

    # Parse each URL
    result = []
    for url in url_list:
        server_name, transport_type, parsed_url = parse_server_url(url)
        
        # Apply HuggingFace authentication if appropriate
        final_headers = add_hf_auth_header(parsed_url, headers)
        
        result.append((server_name, transport_type, parsed_url, final_headers))

    return result


def generate_server_configs(
    parsed_urls: List[Tuple[str, Literal["http", "sse"], str, Dict[str, str] | None]],
) -> Dict[str, Dict[str, str | Dict[str, str]]]:
    """
    Generate server configurations from parsed URLs.

    Args:
        parsed_urls: List of tuples containing (server_name, transport_type, url, headers)

    Returns:
        Dictionary of server configurations
    """
    server_configs = {}
    # Keep track of server name occurrences to handle collisions
    name_counts = {}

    for server_name, transport_type, url, headers in parsed_urls:
        # Handle name collisions by adding a suffix
        final_name = server_name
        if server_name in server_configs:
            # Initialize counter if we haven't seen this name yet
            if server_name not in name_counts:
                name_counts[server_name] = 1

            # Generate a new name with suffix
            suffix = name_counts[server_name]
            final_name = f"{server_name}_{suffix}"
            name_counts[server_name] += 1

            # Ensure the new name is also unique
            while final_name in server_configs:
                suffix = name_counts[server_name]
                final_name = f"{server_name}_{suffix}"
                name_counts[server_name] += 1

        config = {
            "transport": transport_type,
            "url": url,
        }

        # Add headers if provided
        if headers:
            config["headers"] = headers

        server_configs[final_name] = config

    return server_configs
