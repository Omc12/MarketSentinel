class NewsAPIError(Exception):
    """Custom exception for news API errors."""
    pass
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
    title: str
    summary: str
    source: str
    url: str
    published_at: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None

    @property
    def full_text(self) -> str:
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

# Indian stock ticker to company name mapping for better news search
INDIAN_STOCK_MAPPING = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys",
    "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "ITC": "ITC Limited",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "WIPRO": "Wipro",
    "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharmaceutical",
    "ADANIGREEN": "Adani Green Energy",
    "ADANIPORTS": "Adani Ports",
    "ONGC": "Oil and Natural Gas Corporation",
    "NTPC": "NTPC Limited",
    "POWERGRID": "Power Grid Corporation",
    "ULTRACEMCO": "UltraTech Cement",
    "TECHM": "Tech Mahindra",
    "BAJFINANCE": "Bajaj Finance",
    "TITAN": "Titan Company",
    "NESTLEIND": "Nestle India",
    "ASIANPAINT": "Asian Paints",
    "DIVISLAB": "Divi's Laboratories",
    "HEROMOTOCO": "Hero MotoCorp",
    "TATSILV": "Tata Elxsi"
}


def fetch_stock_news(
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

    # Map Indian ticker to company name for better search results
    ticker_upper = ticker.upper()
    query = INDIAN_STOCK_MAPPING.get(ticker_upper, ticker)
    
    base_url = "https://newsdata.io/api/1/news"
    # Free tier only supports basic parameters (q, language, apikey)
    params = {
        "apikey": api_key,
        "q": query,
        "language": "en"
    }
    articles = []
    
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
        return articles
    
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
            if len(articles) >= limit:
                break
    
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
