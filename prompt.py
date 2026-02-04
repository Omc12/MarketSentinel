"""
Prompts and templates for grounded signal generation.
"""

import json
from schema import OUTPUT_SCHEMA


# System prompt for grounded generation
SYSTEM_PROMPT = """You are a financial news analyst that extracts structured signals from stock news articles. Your task is to analyze the provided news context and generate a structured JSON signal.

CRITICAL GROUNDING RULES:
1. Use ONLY information explicitly stated in the provided news context
2. Do NOT invent, assume, or infer facts not present in the articles
3. Do NOT use any external knowledge about the stock or company
4. If information is unclear, conflicting, or insufficient, lower your confidence score
5. Every claim in your output MUST be traceable to a specific article in the context
6. If the context contains no useful information, set confidence to 0.0 and add uncertainty flags

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON matching the exact schema provided
- No markdown, no explanations, no additional text
- Just the raw JSON object

CONFIDENCE GUIDELINES:
- 0.8-1.0: Strong, consistent signals across multiple articles
- 0.5-0.8: Moderate signals with some uncertainty or mixed information
- 0.2-0.5: Weak signals, limited information, or conflicting reports
- 0.0-0.2: Very low confidence, insufficient or contradictory data

SENTIMENT STRENGTH GUIDELINES:
- 0.8-1.0: Very strong sentiment indicators (major earnings beat/miss, significant price moves, major announcements)
- 0.5-0.8: Moderate sentiment signals
- 0.2-0.5: Mild sentiment indicators
- 0.0-0.2: Neutral or very weak sentiment"""


# User prompt template
USER_PROMPT_TEMPLATE = """Analyze the following news context for {ticker} and generate a structured signal.

{context}

Generate a JSON signal with the following exact structure:
{schema}

REMEMBER:
- Base ALL claims ONLY on the provided articles above
- Include evidence linking each major claim to its source article
- Set appropriate confidence based on information quality
- Return ONLY the JSON object, nothing else"""


def get_system_prompt() -> str:
    """Get the system prompt for signal generation."""
    return SYSTEM_PROMPT


def get_user_prompt(ticker: str, context: str) -> str:
    """
    Generate the user prompt with context and schema.
    
    Args:
        ticker: Stock ticker symbol
        context: Formatted news context
        
    Returns:
        Formatted user prompt
    """
    schema_str = json.dumps(OUTPUT_SCHEMA, indent=2)
    
    return USER_PROMPT_TEMPLATE.format(
        ticker=ticker.upper(),
        context=context,
        schema=schema_str
    )


def get_example_output() -> str:
    """Get an example of expected output for few-shot prompting."""
    example = {
        "overall_sentiment": "bullish",
        "sentiment_strength": 0.75,
        "confidence": 0.85,
        "bullish_drivers": [
            "Strong Q4 earnings beat analyst expectations",
            "New AI product announcements received positive reception"
        ],
        "bearish_risks": [
            "Increased regulatory scrutiny on tech sector"
        ],
        "key_events": [
            "Q4 2024 earnings release showing revenue growth",
            "AI feature announcements for upcoming product line"
        ],
        "uncertainty_flags": [
            "Regulatory outcome uncertain"
        ],
        "time_horizon": "short_term",
        "evidence": [
            {
                "claim": "Strong Q4 earnings beat analyst expectations",
                "source": "Article 1 - Reuters: Apple Reports Record Q4 Earnings"
            },
            {
                "claim": "New AI product announcements received positive reception",
                "source": "Article 2 - Bloomberg: Apple Stock Rises on AI Announcements"
            }
        ]
    }
    return json.dumps(example, indent=2)


# Fallback prompt for retry on JSON parse failure
RETRY_PROMPT = """Your previous response was not valid JSON. Please try again.

Return ONLY a valid JSON object with this exact structure, no other text:
{
  "overall_sentiment": "bullish | bearish | neutral",
  "sentiment_strength": <number 0.0-1.0>,
  "confidence": <number 0.0-1.0>,
  "bullish_drivers": [<list of strings>],
  "bearish_risks": [<list of strings>],
  "key_events": [<list of strings>],
  "uncertainty_flags": [<list of strings>],
  "time_horizon": "short_term | medium_term | long_term",
  "evidence": [{"claim": "<string>", "source": "<string>"}]
}

Return ONLY the JSON object, nothing else."""


def get_retry_prompt() -> str:
    """Get the retry prompt for JSON parse failures."""
    return RETRY_PROMPT


# Low confidence fallback template
LOW_CONFIDENCE_SIGNAL = {
    "overall_sentiment": "neutral",
    "sentiment_strength": 0.0,
    "confidence": 0.0,
    "bullish_drivers": [],
    "bearish_risks": [],
    "key_events": [],
    "uncertainty_flags": [
        "Insufficient news data to generate reliable signal",
        "Model could not produce valid analysis"
    ],
    "time_horizon": "short_term",
    "evidence": []
}


def get_low_confidence_signal() -> dict:
    """Get a low confidence signal for error cases."""
    return LOW_CONFIDENCE_SIGNAL.copy()


if __name__ == "__main__":
    # Test prompts
    print("=== System Prompt ===")
    print(get_system_prompt()[:500] + "...")
    print("\n=== Example Output ===")
    print(get_example_output())
