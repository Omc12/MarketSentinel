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

    
        ticker: str,
        days: int = 14,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch recent news for a stock ticker using Newsdata.io API.
        Supports both US and Indian stocks (by company name or ticker).
        Args:
            ticker: Stock ticker symbol or company name (e.g., 'AAPL', 'RELIANCE')
            days: Number of days to look back (default 14)
            limit: Maximum number of articles to return
        Returns:
            List of NewsArticle objects
        Raises:
            NewsAPIError: If API call fails or returns an error
        """
        api_key = os.getenv("NEWSDATA_API_KEY")
        if not api_key:
            raise NewsAPIError("Newsdata.io API key not configured. Please set NEWSDATA_API_KEY in .env file.")

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        # Newsdata.io expects YYYY-MM-DD format
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")

        # Use ticker or company name as query
        query = ticker

        base_url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": api_key,
            "q": query,
            "language": "en",
            "from_date": from_date,
            "to_date": to_date,
            "page": 0,
            "country": "us,in",
            "category": "business"
        }

        articles = []
        fetched = 0
        page = 0
        while fetched < limit:
            params["page"] = page
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:
                raise NewsAPIError(f"Failed to fetch news: {str(e)}")

            if data.get("status") != "success":
                raise NewsAPIError(f"API Error: {data.get('message', 'Unknown error')}")

            news_list = data.get("results", [])
            if not news_list:
                break

            for item in news_list:
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("description", ""),
                    source=item.get("source_id", "Unknown"),
                    url=item.get("link", ""),
                    published_at=item.get("pubDate", "")
                )
                if article.title and article.summary:
                    articles.append(article)
                    fetched += 1
                    if fetched >= limit:
                        break
            page += 1
        return articles
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
