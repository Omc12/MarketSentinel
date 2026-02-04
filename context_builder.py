"""
Context builder for formatting retrieved documents for LLM consumption.
"""

from typing import List
from fetch_news import NewsArticle, format_article_for_context
from retrieval import RetrievalResult


def build_context(
    results: List[RetrievalResult],
    ticker: str,
    max_context_length: int = 8000
) -> str:
    """
    Build formatted context from retrieved articles for LLM input.
    
    Args:
        results: List of retrieval results
        ticker: Stock ticker symbol
        max_context_length: Maximum character length for context
        
    Returns:
        Formatted context string
    """
    if not results:
        return f"No recent news articles found for {ticker}."
    
    context_parts = []
    current_length = 0
    
    # Header
    header = f"=== Recent News for {ticker.upper()} ===\n\n"
    context_parts.append(header)
    current_length += len(header)
    
    # Add each article
    for i, result in enumerate(results):
        article_text = format_article_for_context(result.article, i)
        
        # Check if adding this article would exceed limit
        if current_length + len(article_text) > max_context_length:
            # Add truncation notice
            context_parts.append(f"\n[{len(results) - i} additional articles truncated due to length]\n")
            break
        
        context_parts.append(article_text)
        context_parts.append("\n")
        current_length += len(article_text) + 1
    
    # Footer
    footer = f"\n=== End of News Context ({len(context_parts) - 2} articles) ===\n"
    context_parts.append(footer)
    
    return "".join(context_parts)


def build_context_from_articles(
    articles: List[NewsArticle],
    ticker: str,
    max_context_length: int = 8000
) -> str:
    """
    Build formatted context directly from articles (without retrieval scores).
    
    Args:
        articles: List of news articles
        ticker: Stock ticker symbol
        max_context_length: Maximum character length for context
        
    Returns:
        Formatted context string
    """
    if not articles:
        return f"No recent news articles found for {ticker}."
    
    context_parts = []
    current_length = 0
    
    # Header
    header = f"=== Recent News for {ticker.upper()} ===\n\n"
    context_parts.append(header)
    current_length += len(header)
    
    # Add each article
    for i, article in enumerate(articles):
        article_text = format_article_for_context(article, i)
        
        # Check if adding this article would exceed limit
        if current_length + len(article_text) > max_context_length:
            context_parts.append(f"\n[{len(articles) - i} additional articles truncated due to length]\n")
            break
        
        context_parts.append(article_text)
        context_parts.append("\n")
        current_length += len(article_text) + 1
    
    # Footer
    footer = f"\n=== End of News Context ({len(context_parts) - 2} articles) ===\n"
    context_parts.append(footer)
    
    return "".join(context_parts)


def format_sources_summary(results: List[RetrievalResult]) -> str:
    """
    Create a summary of sources used for transparency.
    
    Args:
        results: List of retrieval results
        
    Returns:
        Formatted sources summary
    """
    if not results:
        return "No sources available."
    
    lines = ["Sources used:"]
    for i, result in enumerate(results):
        article = result.article
        date_str = article.published_at[:8] if article.published_at else "Unknown"
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        lines.append(f"  {i+1}. {article.source} ({date_str}): {article.title[:60]}...")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test context building
    from fetch_news import NewsArticle
    from retrieval import RetrievalResult
    
    articles = [
        NewsArticle(
            title="Apple Reports Record Q4 Earnings",
            summary="Apple Inc. announced record-breaking fourth quarter earnings, beating analyst expectations with strong iPhone sales. The company reported revenue of $119.6 billion.",
            source="Reuters",
            url="https://example.com/1",
            published_at="20240115T120000"
        ),
        NewsArticle(
            title="Apple Stock Rises on AI Announcements",
            summary="Shares of Apple climbed 5% following announcements about new AI features coming to iOS. Analysts are optimistic about the company's AI strategy.",
            source="Bloomberg",
            url="https://example.com/2",
            published_at="20240114T100000"
        ),
    ]
    
    results = [
        RetrievalResult(article=articles[0], index=0, score=0.95, retrieval_type="hybrid+mmr"),
        RetrievalResult(article=articles[1], index=1, score=0.87, retrieval_type="hybrid+mmr"),
    ]
    
    context = build_context(results, "AAPL")
    print(context)
    print("\n" + "="*50 + "\n")
    print(format_sources_summary(results))
