#!/usr/bin/env python3
"""
Ollama API Interface - Shared utility for all apps to interact with Ollama via HTTP API.
Replaces direct ollama package usage with HTTP requests to /v1/chat/completions endpoint.
"""
import json
import os
from typing import Generator, Optional

import requests


def call_ollama_api(
    prompt: str,
    model: str,
    base_url: Optional[str] = None,
    stream: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Call Ollama LLM using the /v1/chat/completions API endpoint.
    
    Args:
        prompt: The prompt text to send to the LLM
        model: The model name (e.g., "llama3.2")
        base_url: Base URL for Ollama (defaults to OLLAMA_URL env var or http://0.0.0.0:11434)
        stream: Whether to use streaming mode (default: False)
        system_prompt: Optional system prompt to set context
    
    Returns:
        The response text from the LLM
    """
    if base_url is None:
        base_url = os.getenv("OLLAMA_URL", "http://0.0.0.0:11434")
    
    try:
        # Construct the API endpoint URL
        api_url = f"{base_url}/v1/chat/completions"
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # Make the HTTP POST request
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5 minute timeout for long responses
            stream=stream  # Enable streaming for requests library
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        if stream:
            # Handle streaming response (Server-Sent Events format)
            full_content = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    # SSE format: "data: {...}"
                    if line_str.startswith("data: "):
                        json_str = line_str[6:]  # Remove "data: " prefix
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(json_str)
                            # Extract content from delta
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                        except json.JSONDecodeError:
                            # Skip invalid JSON chunks
                            continue
            return full_content
        else:
            # Handle non-streaming response
            result = response.json()
            
            # Extract the message content from the response
            # API response format: {"choices": [{"message": {"content": "..."}}]}
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "No response from LLM"
            
    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama API: {e}"
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return f"Error parsing Ollama API response: {e}"


def stream_ollama_api(
    prompt: str,
    model: str,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream LLM response from Ollama using the /v1/chat/completions API endpoint.
    
    Args:
        prompt: The prompt text to send to the LLM
        model: The model name (e.g., "llama3.2")
        base_url: Base URL for Ollama (defaults to OLLAMA_URL env var or http://0.0.0.0:11434)
        system_prompt: Optional system prompt to set context
    
    Yields:
        Chunks of the response text as they arrive
    """
    if base_url is None:
        base_url = os.getenv("OLLAMA_URL", "http://0.0.0.0:11434")
    
    try:
        # Construct the API endpoint URL
        api_url = f"{base_url}/v1/chat/completions"
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        # Make the HTTP POST request
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
            stream=True
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Stream the response
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                # SSE format: "data: {...}"
                if line_str.startswith("data: "):
                    json_str = line_str[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(json_str)
                        # Extract content from delta
                        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                            delta = chunk_data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        # Skip invalid JSON chunks
                        continue
                        
    except requests.exceptions.RequestException as e:
        yield f"Error calling Ollama API: {e}"
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        yield f"Error parsing Ollama API response: {e}"
