"""Angelique: news/NLP workflow. INSERT HERE to complete."""


def analyze_news(ticker):
    """Return JSON-compatible analysis for a ticker.

    INSERT HERE (Angelique):
    1. Ingest headlines via data_loader.load_news(ticker).
    2. Preprocess and clean news text.
    3. Classify news categories and sentiment.
    4. Extract relevant companies, events, and dates.
    5. Summarize the findings with source attribution.

    Expected return format:
    {
        "ticker": ticker,
        "sentiment": "positive/neutral/negative/mixed",
        "summary": "...",
        "events": [{"event": "...", "source": "...", "date": "..."}],
        "article_count": 0,
        "limitations": []
    }
    """
    raise NotImplementedError("Angelique: INSERT HERE news NLP workflow")
