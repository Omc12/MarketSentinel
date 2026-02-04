"""
MarketSentinel - Streamlit Application

A product-style RAG system that converts stock news into structured,
grounded JSON signals for ML consumption.
"""

import streamlit as st
from typing import Optional, List
import json

from fetch_news import fetch_stock_news, NewsArticle, NewsAPIError
from retrieval import HybridRetriever, create_retriever, RetrievalResult
from context_builder import build_context, format_sources_summary
from generate_signals import generate_signal_safe
from schema import StockSignal


# Page configuration
st.set_page_config(
    page_title="MarketSentinel",
    page_icon="📈",
    layout="wide"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .bullish {
        color: #28a745;
        font-weight: bold;
    }
    .bearish {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .evidence-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1f77b4;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .driver-item {
        padding: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_sentiment_color(sentiment: str) -> str:
    """Get color class for sentiment."""
    colors = {
        "bullish": "green",
        "bearish": "red",
        "neutral": "gray"
    }
    return colors.get(sentiment, "gray")


def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment."""
    emojis = {
        "bullish": "🟢",
        "bearish": "🔴",
        "neutral": "⚪"
    }
    return emojis.get(sentiment, "⚪")


def display_signal(signal: StockSignal, ticker: str):
    """Display the signal in a formatted way."""
    
    # Sentiment header
    sentiment_emoji = get_sentiment_emoji(signal.overall_sentiment)
    st.markdown(f"## {sentiment_emoji} Signal for {ticker.upper()}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_color = get_sentiment_color(signal.overall_sentiment)
        st.metric(
            "Overall Sentiment",
            signal.overall_sentiment.upper(),
        )
    
    with col2:
        st.metric(
            "Sentiment Strength",
            f"{signal.sentiment_strength:.0%}",
        )
    
    with col3:
        st.metric(
            "Confidence",
            f"{signal.confidence:.0%}",
        )
    
    with col4:
        horizon_display = signal.time_horizon.replace("_", " ").title()
        st.metric(
            "Time Horizon",
            horizon_display,
        )
    
    st.divider()
    
    # Bullish drivers and bearish risks side by side
    col_bull, col_bear = st.columns(2)
    
    with col_bull:
        st.markdown("### 📈 Bullish Drivers")
        if signal.bullish_drivers:
            for driver in signal.bullish_drivers:
                st.markdown(f"- {driver}")
        else:
            st.markdown("*No bullish drivers identified*")
    
    with col_bear:
        st.markdown("### 📉 Bearish Risks")
        if signal.bearish_risks:
            for risk in signal.bearish_risks:
                st.markdown(f"- {risk}")
        else:
            st.markdown("*No bearish risks identified*")
    
    st.divider()
    
    # Key events
    st.markdown("### 📅 Key Events")
    if signal.key_events:
        for event in signal.key_events:
            st.markdown(f"- {event}")
    else:
        st.markdown("*No key events identified*")
    
    # Uncertainty flags (if any)
    if signal.uncertainty_flags:
        st.divider()
        st.markdown("### ⚠️ Uncertainty Flags")
        for flag in signal.uncertainty_flags:
            st.warning(flag)
    
    # Evidence
    st.divider()
    st.markdown("### 📋 Evidence")
    
    if signal.evidence:
        for i, evidence in enumerate(signal.evidence):
            with st.expander(f"Evidence {i+1}: {evidence.claim[:60]}..."):
                st.markdown(f"**Claim:** {evidence.claim}")
                st.markdown(f"**Source:** {evidence.source}")
    else:
        st.markdown("*No evidence provided*")
    
    # Raw JSON output
    with st.expander("🔧 Raw JSON Output"):
        st.json(signal.to_dict())


def display_retrieved_articles(results: List[RetrievalResult]):
    """Display retrieved articles for debugging."""
    st.markdown("### 📰 Retrieved Articles")
    
    for i, result in enumerate(results):
        article = result.article
        with st.expander(f"Article {i+1}: {article.title[:60]}... (Score: {result.score:.3f})"):
            st.markdown(f"**Source:** {article.source}")
            st.markdown(f"**Date:** {article.published_at[:8] if article.published_at else 'Unknown'}")
            st.markdown(f"**Summary:** {article.summary}")
            if article.url:
                st.markdown(f"[Read more]({article.url})")


@st.cache_resource
def get_retriever() -> HybridRetriever:
    """Get cached retriever instance."""
    return create_retriever()


def run_rag_pipeline(ticker: str, days: int = 14) -> tuple:
    """
    Run the full RAG pipeline.
    
    Returns:
        Tuple of (signal, retrieved_results, context, error_message)
    """
    # Step 1: Fetch news
    try:
        with st.spinner("Fetching recent news..."):
            articles = fetch_stock_news(ticker, days=days, limit=50)
    except NewsAPIError as e:
        return None, None, None, str(e)
    
    if not articles:
        return None, None, None, f"No news articles found for {ticker} in the last {days} days."
    
    st.success(f"Found {len(articles)} news articles")
    
    # Step 2: Retrieval
    with st.spinner("Running hybrid retrieval..."):
        retriever = get_retriever()
        retriever.index_articles(articles)
        
        # Create query
        query = f"{ticker} stock price performance news analysis"
        
        # Retrieve with hybrid + MMR
        results = retriever.retrieve(
            query=query,
            initial_top_k=8,
            final_top_k=4,
            mmr_lambda=0.7
        )
    
    if not results:
        return None, None, None, "Retrieval returned no results."
    
    st.success(f"Retrieved top {len(results)} relevant articles")
    
    # Step 3: Build context
    context = build_context(results, ticker)
    
    # Step 4: Generate signal
    with st.spinner("Generating structured signal with Gemini..."):
        signal, error = generate_signal_safe(ticker, context)
    
    if error:
        return None, results, context, f"Generation error: {error}"
    
    return signal, results, context, None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">📈 MarketSentinel</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Convert recent stock news into structured, grounded signals for ML consumption</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        days = st.slider(
            "News lookback (days)",
            min_value=7,
            max_value=14,
            value=14,
            help="Number of days to look back for news articles"
        )
        
        show_debug = st.checkbox(
            "Show debug info",
            value=False,
            help="Show retrieved articles and raw context"
        )
        
        st.divider()
        
        st.markdown("### About")
        st.markdown("""
        This tool uses RAG to analyze stock news:
        
        1. **Fetch** recent news from Alpha Vantage
        2. **Retrieve** relevant articles (hybrid search + MMR)
        3. **Generate** grounded signals with Gemini
        
        The output is ML-friendly JSON with sentiment, drivers, and evidence.
        """)
        
        st.divider()
        
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This tool is for informational purposes only.
        Not financial advice. Do your own research.
        """)
    
    # Main input area
    col_input, col_button = st.columns([3, 1])
    
    with col_input:
        ticker = st.text_input(
            "Enter Stock Ticker",
            placeholder="e.g., AAPL, TSLA, GOOGL",
            help="Enter a valid stock ticker symbol"
        ).strip().upper()
    
    with col_button:
        st.write("")  # Spacing
        st.write("")  # Spacing
        analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
    
    # Process when button is clicked
    if analyze_button:
        if not ticker:
            st.error("Please enter a stock ticker symbol.")
            return
        
        # Validate ticker format (basic check)
        if not ticker.isalpha() or len(ticker) > 5:
            st.error("Please enter a valid ticker symbol (1-5 letters).")
            return
        
        # Run pipeline
        signal, results, context, error = run_rag_pipeline(ticker, days)
        
        if error:
            st.error(f"❌ {error}")
            
            # Still show partial results if available
            if results and show_debug:
                st.divider()
                display_retrieved_articles(results)
            return
        
        st.divider()
        
        # Display signal
        display_signal(signal, ticker)
        
        # Debug info
        if show_debug:
            st.divider()
            st.markdown("## 🔍 Debug Information")
            
            # Retrieved articles
            if results:
                display_retrieved_articles(results)
            
            # Raw context
            if context:
                with st.expander("📝 Raw Context Sent to LLM"):
                    st.text(context)
    
    # Example tickers
    if not ticker:
        st.markdown("---")
        st.markdown("### 💡 Try These Examples")
        
        example_cols = st.columns(5)
        examples = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA"]
        
        for col, example in zip(example_cols, examples):
            with col:
                if st.button(example, use_container_width=True):
                    st.session_state["ticker"] = example
                    st.rerun()


if __name__ == "__main__":
    main()
