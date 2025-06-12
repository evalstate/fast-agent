"""HuggingFace authentication utilities for MCP connections."""

import os
from typing import Dict, Optional
from urllib.parse import urlparse


def is_huggingface_url(url: str) -> bool:
    """
    Check if a URL is a HuggingFace URL that should receive HF_TOKEN authentication.
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL is a HuggingFace URL, False otherwise
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname is None:
            return False
        
        # Check for HuggingFace domains
        if hostname in {"hf.co", "huggingface.co"}:
            return True
            
        # Check for HuggingFace Spaces (*.hf.space)
        # Use endswith to match subdomains like space-name.hf.space
        # but ensure exact match to prevent spoofing like evil.hf.space.com
        if hostname.endswith(".hf.space") and hostname.count(".") >= 2:
            # Additional validation: ensure it's a valid HF Space domain
            # Format should be: {space-name}.hf.space
            parts = hostname.split(".")
            if len(parts) == 3 and parts[-2:] == ["hf", "space"]:
                space_name = parts[0]
                # Validate space name: not empty, not just hyphens/dots, no spaces
                return (len(space_name) > 0 and 
                        space_name != "-" and 
                        not space_name.startswith(".") and
                        not space_name.endswith(".") and
                        " " not in space_name)
            
        return False
    except Exception:
        return False


def get_hf_token_from_env() -> Optional[str]:
    """
    Get the HuggingFace token from the HF_TOKEN environment variable.
    
    Returns:
        The HF_TOKEN value if set, None otherwise
    """
    return os.environ.get("HF_TOKEN")


def should_add_hf_auth(url: str, existing_headers: Optional[Dict[str, str]]) -> bool:
    """
    Determine if HuggingFace authentication should be added to the headers.
    
    Args:
        url: The URL to check
        existing_headers: Existing headers dictionary (may be None)
        
    Returns:
        True if HF auth should be added, False otherwise
    """
    # Only add HF auth if:
    # 1. URL is a HuggingFace URL
    # 2. No existing Authorization header is set
    # 3. HF_TOKEN environment variable is available
    
    if not is_huggingface_url(url):
        return False
        
    if existing_headers and "Authorization" in existing_headers:
        return False
        
    return get_hf_token_from_env() is not None


def add_hf_auth_header(url: str, headers: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Add HuggingFace authentication header if appropriate.
    
    Args:
        url: The URL to check
        headers: Existing headers dictionary (may be None)
        
    Returns:
        Updated headers dictionary with HF auth if appropriate, or original headers
    """
    if not should_add_hf_auth(url, headers):
        return headers
        
    hf_token = get_hf_token_from_env()
    if hf_token is None:
        return headers
        
    # Create new headers dict or copy existing one
    result_headers = dict(headers) if headers else {}
    result_headers["Authorization"] = f"Bearer {hf_token}"
    
    return result_headers