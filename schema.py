"""
Output schema definitions for Stock News RAG Signals.
"""

from dataclasses import dataclass, field
from typing import List, Literal
import json


@dataclass
class Evidence:
    """Evidence linking a claim to its source."""
    claim: str
    source: str

    def to_dict(self) -> dict:
        return {"claim": self.claim, "source": self.source}


@dataclass
class StockSignal:
    """Structured signal output from the RAG pipeline."""
    overall_sentiment: Literal["bullish", "bearish", "neutral"]
    sentiment_strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    bullish_drivers: List[str] = field(default_factory=list)
    bearish_risks: List[str] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)
    uncertainty_flags: List[str] = field(default_factory=list)
    time_horizon: Literal["short_term", "medium_term", "long_term"] = "short_term"
    evidence: List[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_sentiment": self.overall_sentiment,
            "sentiment_strength": self.sentiment_strength,
            "confidence": self.confidence,
            "bullish_drivers": self.bullish_drivers,
            "bearish_risks": self.bearish_risks,
            "key_events": self.key_events,
            "uncertainty_flags": self.uncertainty_flags,
            "time_horizon": self.time_horizon,
            "evidence": [e.to_dict() for e in self.evidence]
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "StockSignal":
        """Create StockSignal from dictionary."""
        evidence_list = [
            Evidence(claim=e.get("claim", ""), source=e.get("source", ""))
            for e in data.get("evidence", [])
        ]
        return cls(
            overall_sentiment=data.get("overall_sentiment", "neutral"),
            sentiment_strength=float(data.get("sentiment_strength", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            bullish_drivers=data.get("bullish_drivers", []),
            bearish_risks=data.get("bearish_risks", []),
            key_events=data.get("key_events", []),
            uncertainty_flags=data.get("uncertainty_flags", []),
            time_horizon=data.get("time_horizon", "short_term"),
            evidence=evidence_list
        )


# JSON Schema for validation and prompting
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_sentiment": {
            "type": "string",
            "enum": ["bullish", "bearish", "neutral"]
        },
        "sentiment_strength": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "bullish_drivers": {
            "type": "array",
            "items": {"type": "string"}
        },
        "bearish_risks": {
            "type": "array",
            "items": {"type": "string"}
        },
        "key_events": {
            "type": "array",
            "items": {"type": "string"}
        },
        "uncertainty_flags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "time_horizon": {
            "type": "string",
            "enum": ["short_term", "medium_term", "long_term"]
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "source": {"type": "string"}
                },
                "required": ["claim", "source"]
            }
        }
    },
    "required": [
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
}


def get_empty_signal() -> StockSignal:
    """Return an empty/default signal for error cases."""
    return StockSignal(
        overall_sentiment="neutral",
        sentiment_strength=0.0,
        confidence=0.0,
        bullish_drivers=[],
        bearish_risks=[],
        key_events=[],
        uncertainty_flags=["Unable to generate signal - no data or error occurred"],
        time_horizon="short_term",
        evidence=[]
    )
