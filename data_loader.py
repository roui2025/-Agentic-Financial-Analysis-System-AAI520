"""Load market data and news for a stock ticker from Yahoo Finance.

These functions only fetch data. Cleaning and analysis happen downstream
(news preprocessing, market analysis, and the agents that call them).
"""

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

# Fields pulled from Ticker.info for a quick company snapshot.
KEY_METRIC_FIELDS = [
    "longName",
    "sector",
    "industry",
    "currentPrice",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "trailingEps",
    "dividendYield",
    "beta",
    "profitMargins",
    "operatingMargins",
    "revenueGrowth",
    "earningsGrowth",
    "debtToEquity",
    "returnOnEquity",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "fiftyDayAverage",
    "twoHundredDayAverage",
    "recommendationKey",
    "targetMeanPrice",
]

NEWS_COLUMNS = ["title", "summary", "publisher", "published", "url"]


def load_prices(ticker, period="1y", interval="1d"):
    """Return daily OHLCV price history for a ticker.

    Raises ValueError if Yahoo Finance returns no data (for example,
    an invalid or delisted symbol).
    """
    prices = yf.Ticker(ticker).history(period=period, interval=interval)
    if prices.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'.")
    return prices


def load_financials(ticker):
    """Return annual financial statements as a dict of DataFrames.

    Keys: 'income_statement', 'balance_sheet', 'cash_flow'. Each
    DataFrame has line items as rows and fiscal year-end dates as columns.
    """
    stock = yf.Ticker(ticker)
    return {
        "income_statement": stock.income_stmt,
        "balance_sheet": stock.balance_sheet,
        "cash_flow": stock.cashflow,
    }


def load_key_metrics(ticker):
    """Return a dict of key valuation and performance metrics.

    Fields Yahoo Finance does not report for the ticker are set to None.
    Raises ValueError if it reports none of them.
    """
    info = yf.Ticker(ticker).info
    metrics = {field: info.get(field) for field in KEY_METRIC_FIELDS}
    if all(value is None for value in metrics.values()):
        raise ValueError(f"No metrics found for ticker '{ticker}'.")
    metrics["ticker"] = ticker.upper()
    return metrics


def load_news(ticker, count=20):
    """Return recent news headlines for a ticker as a DataFrame.

    Columns: title, summary, publisher, published (UTC), url. Tries the
    ticker news feed first and falls back to Yahoo Finance search, since
    the feed sometimes comes back empty.
    """
    articles = yf.Ticker(ticker).get_news(count=count)
    if not articles:
        articles = yf.Search(ticker, news_count=count).news
    rows = [_parse_article(article) for article in articles]
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def load_all(ticker, period="1y", news_count=20):
    """Return prices, financials, key metrics, and news in one dict."""
    return {
        "ticker": ticker.upper(),
        "prices": load_prices(ticker, period=period),
        "financials": load_financials(ticker),
        "key_metrics": load_key_metrics(ticker),
        "news": load_news(ticker, count=news_count),
    }


def _parse_article(article):
    """Flatten one Yahoo Finance news item into a single row.

    Handles both the nested format ('content' key) returned by the
    ticker feed and the flat format returned by search.
    """
    if "content" in article:
        content = article["content"]
        return {
            "title": content.get("title"),
            "summary": content.get("summary"),
            "publisher": (content.get("provider") or {}).get("displayName"),
            "published": pd.to_datetime(content.get("pubDate"), utc=True),
            "url": (content.get("canonicalUrl") or {}).get("url"),
        }
    timestamp = article.get("providerPublishTime")
    published = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if timestamp
        else None
    )
    return {
        "title": article.get("title"),
        "summary": article.get("summary"),
        "publisher": article.get("publisher"),
        "published": published,
        "url": article.get("link"),
    }


if __name__ == "__main__":
    data = load_all("AAPL")
    print(data["prices"].tail())
    print(data["key_metrics"])
    print(data["news"].head())
