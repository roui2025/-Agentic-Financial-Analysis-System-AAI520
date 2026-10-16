"""Angelique: news/NLP workflow."""

"""
Financial News NLP Agent

Retrieves, preprocesses, classifies, and summarizes
financial news using Yahoo Finance and Llama 3.2.
"""

import json
import requests
import pandas as pd

from data_loader import load_news

def preprocess_news(news_df):
    """Clean and standardize financial news for NLP analysis."""

    df = news_df.copy()

    # Remove articles without titles
    df = df.dropna(subset=["title"])

    # Remove empty titles
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""]

    # Remove duplicate headlines
    df = df.drop_duplicates(subset=["title"])

    # Standardize publication dates
    df["published"] = pd.to_datetime(
        df["published"], errors="coerce", utc=True
    )

    # Replace missing summaries with empty strings
    df["summary"] = df["summary"].fillna("")

    # Combine available text for analysis
    df["text"] = (
        df["title"] + " " + df["summary"]
    ).str.strip()

    # Reset index
    df = df.reset_index(drop=True)

    return df


def classify_news(articles, ticker):
    """Classify financial news using Llama 3.2."""

    classifications = []

    for _, article in articles.iterrows():

        prompt = f"""
        You are a financial news analyst.

        Analyze the following headline for ticker {ticker}.

        Headline: {article['text']}

        Return ONLY a JSON object with these fields:

        {{
            "relevant": true,
            "category": "earnings",
            "sentiment": "neutral",
            "reason": "Brief explanation"
        }}

        Rules:
        - relevant must be true or false.
        - category must be one of:
          earnings, product, regulation, market,
          leadership, partnerships, other.
        - sentiment must be positive, negative, or neutral.
        - Judge sentiment toward the target company,
          not the emotional tone of the headline.
        - If the headline does not provide enough
          evidence, use neutral.
        - Do not invent facts beyond the headline.
        """

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=120
        )

        response.raise_for_status()
        result = response.json()["response"]

        try:
            classification = json.loads(result)

            classification["title"] = article["title"]
            classification["publisher"] = article["publisher"]
            classification["published"] = (
                article["published"].isoformat()
                if pd.notna(article["published"])
                else None
            )
            classification["url"] = article["url"]

            classifications.append(classification)

        except (json.JSONDecodeError, TypeError):
            print(f"Could not classify: {article['title']}")

    return classifications


def extract_events(classified_articles, ticker):
    """Extract factual events from relevant financial headlines."""

    events = []

    for article in classified_articles:

        if not article.get("relevant", False):
            continue

        prompt = f"""
        You are a financial information extraction agent.

        Target ticker: {ticker}
        Headline: {article['title']}

        Extract only concrete events explicitly supported
        by the headline.

        Return ONLY a JSON object in this format:

        {{
            "events": [
                {{
                    "event": "Brief factual description",
                    "companies": ["Company name"],
                    "event_date": null
                }}
            ]
        }}

        Rules:
        - Do not invent financial figures, outcomes, or dates.
        - Do not treat predictions or opinions as confirmed events.
        - If a headline describes an opinion, label it as an opinion.
        - If no concrete event can be identified, return
          an empty events list.
        - Do not assume the article publication date is
          the event date.
        - Relative dates such as "Friday" should remain
          relative unless the exact date is explicitly stated.
        """

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=120
        )

        response.raise_for_status()

        result = json.loads(response.json()["response"])

        for event in result.get("events", []):
            if not isinstance(event, dict):
                continue

            events.append({
                "event": event.get("event", ""),
                "companies": event.get("companies", []),
                "date": event.get("event_date"),
                "source": article.get("publisher"),
                "published": article.get("published"),
                "url": article.get("url")
            })

    return events


def filter_events(events, ticker):
    """Keep events explicitly related to the target company."""

    company_aliases = {
        "AAPL": ["apple", "iphone", "ipad", "macbook"],
        "MSFT": ["microsoft", "windows", "azure", "xbox"],
        "NVDA": ["nvidia", "geforce", "cuda"],
        "GOOGL": ["google", "alphabet", "youtube"]
    }

    aliases = company_aliases.get(ticker.upper(), [ticker.lower()])

    filtered = []

    for event in events:
        companies = " ".join(event.get("companies", []))
        event_text = event.get("event", "")

        combined_text = f"{companies} {event_text}".lower()

        if any(alias in combined_text for alias in aliases):
            filtered.append(event)

    return filtered


def summarize_news(classified_articles, filtered_events, ticker):
    """Summarize financial headlines without unsupported claims."""

    relevant_articles = [
        article for article in classified_articles
        if article.get("relevant") is True
    ]

    if not relevant_articles:
        return "No relevant financial news was found for this ticker."

    evidence = {
        "articles": [
            {
                "title": article["title"],
                "publisher": article["publisher"],
                "published": article["published"],
                "url": article["url"]
            }
            for article in relevant_articles
        ],
        "events": filtered_events
    }

    prompt = f"""
    You are a cautious financial news research analyst.

    Target ticker: {ticker}

    EVIDENCE:
    {json.dumps(evidence, indent=2)}

    TASK:
    Write a concise 2-4 sentence news summary.

    STRICT RULES:
    1. Use ONLY facts explicitly stated in the evidence.
    2. Attribute every factual claim to its publisher.
    3. Distinguish opinions from confirmed developments.
    4. Do NOT predict stock prices, revenue, or investor returns.
    5. Do NOT recommend buying or selling stocks.
    6. Do NOT introduce facts not present in the headlines.
    7. Do NOT discuss unrelated companies.
    8. Do NOT turn questions in headlines into factual claims.
    9. If the evidence is limited, acknowledge that only
       headlines were available.
    10. Every sentence must directly concern the target
    company. Do not mention other companies even if
    they appear in the original headline.

    GOOD EXAMPLE:
    "According to Motley Fool, Apple's iPhone Duo preorders
    are scheduled to begin Friday. The headline questions
    whether the product could influence investor interest,
    but does not establish a financial outcome."

    BAD EXAMPLE:
    "Apple stock will rise because its new iPhone
    will increase sales."

    Return ONLY the summary.
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip().strip('"')


def analyze_news(ticker):
    """
    Run the complete financial news NLP workflow.

    Pipeline:
    Ingest -> Preprocess -> Classify -> Extract -> Summarize
    """

    ticker = ticker.strip().upper()

    # 1. Ingest financial news
    raw_news = load_news(ticker, count=1)

    # 2. Preprocess articles
    clean_news = preprocess_news(raw_news)

    if clean_news.empty:
        return {
            "ticker": ticker,
            "sentiment": "neutral",
            "summary": "No news articles were available.",
            "events": [],
            "article_count": 0,
            "limitations": ["No news articles retrieved."]
        }

    # 3. Classify articles
    classified = classify_news(clean_news, ticker)

    # 4. Extract financial events
    extracted = extract_events(classified, ticker)

    # 5. Filter unrelated events
    relevant_events = filter_events(extracted, ticker)

    # 6. Summarize findings
    summary = summarize_news(
        classified, relevant_events, ticker
    )

    # 7. Aggregate relevant article sentiment
    relevant_sentiments = [
        article.get("sentiment", "neutral")
        for article in classified
        if article.get("relevant") is True
    ]

    non_neutral = [
        s for s in relevant_sentiments
        if s in ("positive", "negative")
    ]

    if not non_neutral:
        overall_sentiment = "neutral"
    elif len(set(non_neutral)) > 1:
        overall_sentiment = "mixed"
    else:
        overall_sentiment = non_neutral[0]

    # 8. Return coordinator-compatible output
    return {
        "ticker": ticker,
        "sentiment": overall_sentiment,
        "summary": summary,
        "events": relevant_events,
        "article_count": len(classified),
        "limitations": [
            "Analysis is based on headlines and available "
            "summaries, not verified full-text articles.",
            "Sentiment is generated by an LLM and may "
            "contain classification errors."
        ]
    }
