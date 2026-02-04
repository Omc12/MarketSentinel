"""
Fetch stock news from Alpha Vantage News & Sentiment API.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class NewsArticle:
    """Represents a single news article."""
    title: str
    summary: str
    source: str
    url: str
    published_at: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    
    @property
    def full_text(self) -> str:
        """Combine title and summary for embedding/retrieval."""
        return f"{self.title}\n\n{self.summary}"
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label
        }


class NewsAPIError(Exception):
    """Custom exception for news API errors."""
    pass


def fetch_stock_news(
    ticker: str,
    days: int = 14,
    limit: int = 50
) -> List[NewsArticle]:
    """
    Fetch recent news for a stock ticker from Alpha Vantage.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        days: Number of days to look back (7-14 recommended)
        limit: Maximum number of articles to return
        
    Returns:
        List of NewsArticle objects
        
    Raises:
        NewsAPIError: If API call fails or returns an error
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key or api_key == "your_alpha_vantage_api_key_here":
        raise NewsAPIError("Alpha Vantage API key not configured. Please set ALPHA_VANTAGE_API_KEY in .env file.")
    
    # Calculate time range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for API (YYYYMMDDTHHMM format)
    time_from = start_date.strftime("%Y%m%dT0000")
    time_to = end_date.strftime("%Y%m%dT2359")
    
    # Build API URL
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker.upper(),
        "time_from": time_from,
        "time_to": time_to,
        "limit": limit,
        "apikey": api_key
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise NewsAPIError(f"Failed to fetch news: {str(e)}")
    
    # Check for API errors
    if "Error Message" in data:
        raise NewsAPIError(f"API Error: {data['Error Message']}")
    
    if "Information" in data:
        # Rate limit or other info message
        raise NewsAPIError(f"API Info: {data['Information']}")
    
    if "Note" in data:
        # API call frequency message
        raise NewsAPIError(f"API Rate Limit: {data['Note']}")
    
    # Parse articles
    articles = []
    feed = data.get("feed", [])
    
    if not feed:
        return articles
    
    for item in feed:
        # Extract ticker-specific sentiment if available
        ticker_sentiment = None
        ticker_upper = ticker.upper()
        
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker_upper:
                ticker_sentiment = ts
                break
        
        article = NewsArticle(
            title=item.get("title", ""),
            summary=item.get("summary", ""),
            source=item.get("source", "Unknown"),
            url=item.get("url", ""),
            published_at=item.get("time_published", ""),
            sentiment_score=float(ticker_sentiment.get("ticker_sentiment_score", 0)) if ticker_sentiment else None,
            sentiment_label=ticker_sentiment.get("ticker_sentiment_label") if ticker_sentiment else None
        )
        
        # Only include articles with meaningful content
        if article.title and article.summary:
            articles.append(article)
    
    return articles


def format_article_for_context(article: NewsArticle, index: int) -> str:
    """Format a single article for inclusion in LLM context."""
    date_str = article.published_at[:8] if article.published_at else "Unknown date"
    if len(date_str) == 8:
        # Format YYYYMMDD to YYYY-MM-DD
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    return f"""[Article {index + 1}]
Source: {article.source}
Date: {date_str}
Title: {article.title}
Summary: {article.summary}
"""


if __name__ == "__main__":
    # Test the news fetching
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching news for {ticker}...")
    
    try:
        articles = fetch_stock_news(ticker, days=7, limit=10)
        print(f"Found {len(articles)} articles\n")
        
        for i, article in enumerate(articles):
            print(f"--- Article {i+1} ---")
            print(f"Title: {article.title}")
            print(f"Source: {article.source}")
            print(f"Date: {article.published_at}")
            print(f"Sentiment: {article.sentiment_label} ({article.sentiment_score})")
            print()
    except NewsAPIError as e:
        print(f"Error: {e}")
