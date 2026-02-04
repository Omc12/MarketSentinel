"""
Signal generation using Gemini with grounded prompting.
"""

import os
import json
import re
from typing import Optional, Tuple
from dotenv import load_dotenv

import warnings
# Suppress the deprecation warning for now
warnings.filterwarnings('ignore', message='All support for the `google.generativeai` package has ended')
import google.generativeai as genai

from schema import StockSignal, get_empty_signal
from prompt import (
    get_system_prompt,
    get_user_prompt,
    get_retry_prompt,
    get_low_confidence_signal
)

load_dotenv()


class GenerationError(Exception):
    """Custom exception for generation errors."""
    pass


def configure_gemini() -> None:
    """Configure the Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise GenerationError("Gemini API key not configured. Please set GEMINI_API_KEY in .env file.")
    
    genai.configure(api_key=api_key)


def extract_json_from_response(text: str) -> Optional[dict]:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        text: Raw response text from LLM
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Clean up the response
    text = text.strip()
    
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1).strip()
    
    # Try to find JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_signal_dict(data: dict) -> Tuple[bool, str]:
    """
    Validate that a dict has the required signal structure.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        "overall_sentiment",
        "sentiment_strength",
        "confidence",
        "bullish_drivers",
        "bearish_risks",
        "key_events",
        "uncertainty_flags",
        "time_horizon",
        "evidence"
    ]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate sentiment
    if data["overall_sentiment"] not in ["bullish", "bearish", "neutral"]:
        return False, f"Invalid overall_sentiment: {data['overall_sentiment']}"
    
    # Validate time horizon
    if data["time_horizon"] not in ["short_term", "medium_term", "long_term"]:
        return False, f"Invalid time_horizon: {data['time_horizon']}"
    
    # Validate numeric fields
    try:
        strength = float(data["sentiment_strength"])
        if not 0 <= strength <= 1:
            return False, f"sentiment_strength must be between 0 and 1: {strength}"
    except (ValueError, TypeError):
        return False, f"Invalid sentiment_strength: {data['sentiment_strength']}"
    
    try:
        confidence = float(data["confidence"])
        if not 0 <= confidence <= 1:
            return False, f"confidence must be between 0 and 1: {confidence}"
    except (ValueError, TypeError):
        return False, f"Invalid confidence: {data['confidence']}"
    
    # Validate list fields
    list_fields = ["bullish_drivers", "bearish_risks", "key_events", "uncertainty_flags", "evidence"]
    for field in list_fields:
        if not isinstance(data[field], list):
            return False, f"{field} must be a list"
    
    return True, ""


def generate_signal(
    ticker: str,
    context: str,
    model_name: str = "models/gemini-2.5-flash",
    max_retries: int = 2
) -> StockSignal:
    """
    Generate structured signal from news context using Gemini.
    
    Args:
        ticker: Stock ticker symbol
        context: Formatted news context
        model_name: Gemini model to use
        max_retries: Maximum retry attempts for JSON parsing
        
    Returns:
        StockSignal object with generated analysis
        
    Raises:
        GenerationError: If generation fails after retries
    """
    configure_gemini()
    
    # Create the model
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
    )
    
    # Build prompts
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(ticker, context)
    
    # Combine for Gemini (which doesn't have separate system prompt in basic API)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Generate response
            if attempt == 0:
                response = model.generate_content(full_prompt)
            else:
                # Retry with additional guidance
                retry_prompt = get_retry_prompt()
                response = model.generate_content(f"{full_prompt}\n\n{retry_prompt}")
            
            # Check for valid response
            if not response.text:
                last_error = "Empty response from model"
                continue
            
            # Extract JSON
            signal_dict = extract_json_from_response(response.text)
            
            if signal_dict is None:
                last_error = f"Failed to parse JSON from response (first 500 chars): {response.text[:500]}..."
                continue
            
            # Validate structure
            is_valid, error_msg = validate_signal_dict(signal_dict)
            
            if not is_valid:
                last_error = f"Invalid signal structure: {error_msg}"
                continue
            
            # Create and return signal
            return StockSignal.from_dict(signal_dict)
            
        except Exception as e:
            last_error = str(e)
            continue
    
    # All retries failed, return low confidence signal
    low_conf = get_low_confidence_signal()
    low_conf["uncertainty_flags"].append(f"Generation error: {last_error}")
    return StockSignal.from_dict(low_conf)


def generate_signal_safe(
    ticker: str,
    context: str,
    model_name: str = "models/gemini-2.5-flash"
) -> Tuple[StockSignal, Optional[str]]:
    """
    Safe wrapper for signal generation that catches all errors.
    
    Args:
        ticker: Stock ticker symbol
        context: Formatted news context
        model_name: Gemini model to use
        
    Returns:
        Tuple of (StockSignal, error_message or None)
    """
    try:
        signal = generate_signal(ticker, context, model_name)
        return signal, None
    except GenerationError as e:
        return get_empty_signal(), str(e)
    except Exception as e:
        return get_empty_signal(), f"Unexpected error: {str(e)}"


if __name__ == "__main__":
    # Test generation
    test_context = """=== Recent News for AAPL ===

[Article 1]
Source: Reuters
Date: 2024-01-15
Title: Apple Reports Record Q4 Earnings
Summary: Apple Inc. announced record-breaking fourth quarter earnings, beating analyst expectations with strong iPhone sales. Revenue reached $119.6 billion, up 2% year over year.

[Article 2]
Source: Bloomberg
Date: 2024-01-14
Title: Apple Stock Rises on AI Announcements
Summary: Shares of Apple climbed 5% following announcements about new AI features coming to iOS. Analysts are optimistic about the company's AI strategy and potential revenue growth.

=== End of News Context (2 articles) ===
"""
    
    print("Generating signal for AAPL...")
    signal, error = generate_signal_safe("AAPL", test_context)
    
    if error:
        print(f"Error: {error}")
    else:
        print("Generated Signal:")
        print(signal.to_json())
