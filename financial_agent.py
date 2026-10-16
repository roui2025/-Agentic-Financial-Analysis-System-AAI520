"""Francisco: market and financial analysis for one ticker.

Loads the data through data_loader.py, calculates return, risk and ratio
metrics, and asks llama3.2 (through local Ollama) to write a short summary
using only those numbers. If the summary includes a number that isn't in
the data, or Ollama is not running, a simple template summary is used
instead so the coordinator always gets a result back.
"""
import math

from data_loader import load_financials, load_key_metrics, load_prices
from evaluator import ask_llm_json, unsupported_numbers

TRADING_DAYS = 252  # trading days in a year, used to annualize volatility
FULL_YEAR_DAYS = 240  # less history than this means under a year of data


def _float(value):
    """Convert to a plain float. Returns None for missing, NaN or infinite values."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def _round(value, digits=2):
    value = _float(value)
    return None if value is None else round(value, digits)


def _pct(part, whole):
    part, whole = _float(part), _float(whole)
    if part is None or not whole:
        return None
    return round(part / whole * 100, 2)


def _billions(value):
    value = _float(value)
    return None if value is None else round(value / 1e9, 2)


def price_metrics(prices):
    """Calculate return, risk and trend metrics from the daily price history."""
    close = prices["Close"].dropna()
    latest = float(close.iloc[-1])
    daily_returns = close.pct_change().dropna()
    drawdown = close / close.cummax() - 1
    high, low = float(prices["High"].max()), float(prices["Low"].min())
    dividends = float(prices["Dividends"].sum()) if "Dividends" in prices else 0.0

    def trailing_return(days):
        if len(close) <= days:
            return None
        start = float(close.iloc[-1 - days])
        return _pct(latest - start, start)

    def moving_average(days):
        return _round(close.tail(days).mean()) if len(close) >= days else None

    return {
        "latest_price": _round(latest),
        "return_1y_pct": _pct(latest - float(close.iloc[0]), float(close.iloc[0])),
        "return_3m_pct": trailing_return(63),
        "return_1m_pct": trailing_return(21),
        "volatility_1y_pct": (_round(daily_returns.std() * math.sqrt(TRADING_DAYS) * 100)
                              if len(daily_returns) > 1 else None),
        "max_drawdown_1y_pct": _round(drawdown.min() * 100),
        "high_52w": _round(high),
        "low_52w": _round(low),
        "pct_below_52w_high": _pct(high - latest, high),
        "sma_50": moving_average(50),
        "sma_200": moving_average(200),
        "dividend_yield_pct": _pct(dividends, latest) if dividends else 0.0,
    }


def price_trend(metrics):
    """Label the past year as 'uptrend', 'downtrend', 'mixed' or 'unknown'.

    This describes what already happened and is not a prediction. An
    uptrend means the price is above its 200-day average and up for the year.
    """
    latest, sma_200 = metrics["latest_price"], metrics["sma_200"]
    yearly = metrics["return_1y_pct"]
    if sma_200 is None or yearly is None:
        return "unknown"
    if latest > sma_200 and yearly > 0:
        return "uptrend"
    if latest < sma_200 and yearly < 0:
        return "downtrend"
    return "mixed"


def _cell(frame, row, column):
    if frame is None or frame.empty or row not in frame.index or column not in frame.columns:
        return None
    return _float(frame.at[row, column])


def fundamental_metrics(financials):
    """Calculate ratios from the most recent annual statements that have revenue.

    Returns an empty dict when there are no statements, which is normal
    for ETFs and funds.
    """
    income = financials.get("income_statement")
    balance = financials.get("balance_sheet")
    cash = financials.get("cash_flow")
    if income is None or income.empty:
        return {}
    years = [year for year in sorted(income.columns, reverse=True)
             if _cell(income, "Total Revenue", year) is not None]
    if not years:
        return {}
    year = years[0]
    prior = years[1] if len(years) > 1 else None

    revenue = _cell(income, "Total Revenue", year)
    prior_revenue = _cell(income, "Total Revenue", prior) if prior is not None else None
    free_cash_flow = _cell(cash, "Free Cash Flow", year)
    if free_cash_flow is None:
        operating = _cell(cash, "Operating Cash Flow", year)
        capex = _cell(cash, "Capital Expenditure", year)
        if operating is not None and capex is not None:
            free_cash_flow = operating + capex  # capex is reported as a negative number
    equity = _cell(balance, "Stockholders Equity", year)
    debt = _cell(balance, "Total Debt", year)

    return {
        "fiscal_year_end": year.date().isoformat(),
        "revenue_bn": _billions(revenue),
        "revenue_growth_pct": (_pct(revenue - prior_revenue, prior_revenue)
                               if prior_revenue else None),
        "net_income_bn": _billions(_cell(income, "Net Income", year)),
        "gross_margin_pct": _pct(_cell(income, "Gross Profit", year), revenue),
        "operating_margin_pct": _pct(_cell(income, "Operating Income", year), revenue),
        "net_margin_pct": _pct(_cell(income, "Net Income", year), revenue),
        "free_cash_flow_bn": _billions(free_cash_flow),
        "debt_to_equity": (_round(debt / equity)
                           if debt is not None and equity and equity > 0 else None),
        "current_ratio": _round(_divide(_cell(balance, "Current Assets", year),
                                        _cell(balance, "Current Liabilities", year))),
    }


def _divide(part, whole):
    return part / whole if part is not None and whole else None


def valuation_metrics(key_metrics):
    """Valuation numbers as Yahoo Finance reports them at the time of the run."""
    return {
        "market_cap_bn": _billions(key_metrics.get("marketCap")),
        "trailing_pe": _round(key_metrics.get("trailingPE")),
        "forward_pe": _round(key_metrics.get("forwardPE")),
        "price_to_book": _round(key_metrics.get("priceToBook")),
        "beta": _round(key_metrics.get("beta")),
    }


def _template_summary(analysis):
    """Build a basic summary straight from the calculated numbers."""
    name = analysis["company"] or analysis["ticker"]
    m, f, v = analysis["metrics"], analysis["fundamentals"], analysis["valuation"]
    period = analysis["period"]
    parts = [f"{name} ({analysis['ticker']}) closed at {m['latest_price']} on "
             f"{analysis['as_of']}"]
    if m["return_1y_pct"] is not None:
        parts[0] += (f", a {m['return_1y_pct']}% change since {period['start']}")
    parts[0] += "."
    if m["volatility_1y_pct"] is not None:
        parts.append(f"Annualized volatility was {m['volatility_1y_pct']}% and the "
                     f"largest drop from a peak was {m['max_drawdown_1y_pct']}%.")
    parts.append(f"The price is {m['pct_below_52w_high']}% below the period high of "
                 f"{m['high_52w']}, and the period low was {m['low_52w']}. The "
                 f"trend over the period is '{analysis['price_trend']}'.")
    if f:
        sentence = f"For the fiscal year ending {f['fiscal_year_end']}"
        facts = []
        if f["revenue_bn"] is not None:
            growth = (f" ({f['revenue_growth_pct']}% vs. the prior year)"
                      if f["revenue_growth_pct"] is not None else "")
            facts.append(f"revenue was {f['revenue_bn']} billion{growth}")
        if f["net_margin_pct"] is not None:
            facts.append(f"net margin was {f['net_margin_pct']}%")
        if f["free_cash_flow_bn"] is not None:
            facts.append(f"free cash flow was {f['free_cash_flow_bn']} billion")
        if facts:
            parts.append(sentence + ", " + ", ".join(facts) + ".")
    if v.get("trailing_pe") is not None:
        parts.append(f"Yahoo Finance reports a trailing P/E of {v['trailing_pe']}.")
    return " ".join(parts)


def _summarize(analysis):
    """Return (summary, note). Tries the LLM first and falls back to the template.

    note is None when the LLM summary was used, otherwise it explains why
    the template was used.
    """
    facts = {key: analysis[key] for key in
             ("ticker", "company", "as_of", "period", "price_trend",
              "metrics", "fundamentals", "valuation")}
    prompt = (
        "You are a careful financial analyst. Write a 3-5 sentence summary "
        "of the DATA for a non-expert. Rules: use ONLY numbers that appear "
        "in DATA, copied as written; do not calculate new numbers. Give the "
        "price date ('as_of') and the 'fiscal_year_end' for statement "
        "figures. Keys ending in _pct are percentages and keys ending in "
        "_bn are billions. Describe the past only: no predictions and no "
        'buy or sell advice. Return ONLY JSON: {"summary": "..."}. '
        f"DATA: {facts}"
    )
    try:
        for _ in range(2):
            summary = str(ask_llm_json(prompt).get("summary", "")).strip()
            bad = unsupported_numbers(summary, facts)
            if summary and not bad:
                return summary, None
            prompt += (f" Your last summary used numbers that are not in DATA: "
                       f"{bad}. Write it again using only numbers from DATA.")
    except (RuntimeError, ValueError) as error:
        return _template_summary(analysis), (
            f"The summary was written from a template because the LLM call failed: {error}")
    return _template_summary(analysis), (
        "The summary was written from a template because the LLM summary kept "
        "using numbers that are not in the data.")


def analyze_financials(ticker):
    """Return JSON-compatible financial analysis for a ticker.

    Raises ValueError if there is no price history, for example with a
    wrong or delisted ticker. The coordinator catches it and saves it as
    this agent's error. Statements and valuation data are optional. If they
    are missing, the analysis still runs and the gap is listed in
    'limitations'.
    """
    ticker = ticker.strip().upper()
    prices = load_prices(ticker, period="1y")
    limitations = [
        "Prices are daily bars from Yahoo Finance, adjusted for splits and "
        "dividends. If the market is open, the latest bar may be an intraday price.",
        "All metrics describe past performance and are not a forecast.",
    ]

    try:
        key_metrics = load_key_metrics(ticker)
    except Exception as error:  # yfinance can raise different kinds of errors here
        key_metrics = {}
        limitations.append(f"Valuation data was unavailable: {error}")
    try:
        fundamentals = fundamental_metrics(load_financials(ticker))
    except Exception as error:
        fundamentals = {}
        limitations.append(f"Financial statements could not be loaded: {error}")

    metrics = price_metrics(prices)
    if len(prices) < FULL_YEAR_DAYS:
        limitations.append(
            f"Only {len(prices)} trading days of history were available, so "
            "the 1-year metrics cover a shorter period.")
    if fundamentals:
        limitations.append(
            "Statement figures come from the latest annual report (fiscal "
            f"year ending {fundamentals['fiscal_year_end']}), not the latest "
            "quarter, and are in the company's reporting currency.")
    elif not any("statements" in note for note in limitations):
        limitations.append("No annual financial statements were available "
                           "(common for ETFs and funds).")
    valuation = valuation_metrics(key_metrics)
    if any(value is not None for value in valuation.values()):
        limitations.append(
            "Valuation multiples (market cap, P/E, price-to-book, beta) are "
            "Yahoo Finance snapshots at run time.")

    analysis = {
        "ticker": ticker,
        "company": key_metrics.get("longName"),
        "sector": key_metrics.get("sector"),
        "industry": key_metrics.get("industry"),
        "as_of": prices.index[-1].date().isoformat(),
        "period": {"start": prices.index[0].date().isoformat(),
                   "end": prices.index[-1].date().isoformat(),
                   "trading_days": len(prices)},
        "price_trend": price_trend(metrics),
        "metrics": metrics,
        "fundamentals": fundamentals,
        "valuation": valuation,
    }
    summary, note = _summarize(analysis)
    if note:
        limitations.append(note)
    return {"ticker": ticker, "summary": summary, **analysis,
            "limitations": limitations}
