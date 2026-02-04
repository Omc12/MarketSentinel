# 

A Retrieval-Augmented Generation (RAG) system that converts recent stock news into structured, grounded JSON signals for downstream ML consumption.

## Features

- **Hybrid Retrieval**: Combines semantic search (sentence embeddings) with keyword search (BM25)
- **MMR Re-ranking**: Reduces redundancy in retrieved documents using Maximal Marginal Relevance
- **Grounded Generation**: Uses Gemini with strict prompting to ensure factual, evidence-based outputs
- **Structured Output**: Returns ML-friendly JSON signals with sentiment, drivers, risks, and evidence

## Output Schema

```json
{
  "overall_sentiment": "bullish | bearish | neutral",
  "sentiment_strength": 0.0,
  "confidence": 0.0,
  "bullish_drivers": [],
  "bearish_risks": [],
  "key_events": [],
  "uncertainty_flags": [],
  "time_horizon": "short_term | medium_term | long_term",
  "evidence": [
    { "claim": "", "source": "" }
  ]
}
```

## Setup

1. Clone the repository:
```bash
cd stock-news-rag-signals
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### API Keys Required

- **Alpha Vantage**: Get a free key at https://www.alphavantage.co/support/#api-key
- **Google Gemini**: Get a key at https://makersuite.google.com/app/apikey

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then:
1. Enter a stock ticker (e.g., AAPL, TSLA, GOOGL)
2. Click "Analyze News"
3. View the structured sentiment signals

## Project Structure

```
stock-news-rag-signals/
├── app.py                  # Streamlit UI entry point
├── fetch_news.py           # Alpha Vantage news fetching
├── retrieval.py            # Hybrid retrieval (semantic + BM25)
├── mmr.py                  # Maximal Marginal Relevance re-ranking
├── context_builder.py      # Context formatting for LLM
├── prompt.py               # System prompts and templates
├── generate_signals.py     # Gemini signal generation
├── schema.py               # Output schema definitions
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md               # This file
```

## Architecture

```
User Input (Ticker)
       ↓
  Fetch News (Alpha Vantage, 7-14 days)
       ↓
  Hybrid Retrieval (Semantic + BM25) → Top 8
       ↓
  MMR Re-ranking → Top 4
       ↓
  Context Builder
       ↓
  Gemini Generation (Grounded)
       ↓
  Structured JSON Signals
```

## Design Philosophy

- **RAG for Signals Only**: Converts unstructured news → structured signals
- **No Price Prediction**: Explicitly out of scope
- **Grounded Generation**: Strict prompting ensures factual outputs
- **ML-Friendly Output**: JSON schema designed for downstream consumption
- **Explainable**: Evidence linking claims to sources

## Limitations

- Alpha Vantage free tier has rate limits (5 calls/minute, 500 calls/day)
- News coverage may vary by ticker
- Gemini responses may occasionally need retry for valid JSON

## License

MIT License
