"""Sentiment data scrapers — collect non-quantitative data from free sources.

Data sources:
  1. yfinance news headlines (per-ticker)
  2. RSS feeds from CNBC, Reuters, MarketWatch, WSJ
  3. FRED economic data (unemployment, GDP, CPI, Fed rate)
  4. SEC EDGAR insider trading (Form 4 filings)
  5. CNN Fear & Greed Index
  6. Geopolitical/conflict keyword detection from news
"""

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import feedparser
import requests
import yfinance

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "models" / "sentiment_cache.db"
DB_PATH.parent.mkdir(exist_ok=True)

# RSS feeds for financial news
RSS_FEEDS = {
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "cnbc_world": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    "cnbc_business": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "marketwatch_top": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "marketwatch_markets": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "reuters_business": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
    "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "wsj_world": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}

# Geopolitical/conflict keywords and their sentiment impact
CONFLICT_KEYWORDS = {
    # Wars and military
    "war": -0.8, "invasion": -0.9, "missile": -0.7, "bombing": -0.8,
    "airstrike": -0.7, "troops": -0.4, "military": -0.3, "nuclear": -0.8,
    "sanctions": -0.5, "ceasefire": 0.4, "peace deal": 0.6, "peace talks": 0.3,
    "escalation": -0.6, "de-escalation": 0.4,
    # Trade conflicts
    "tariff": -0.5, "trade war": -0.7, "trade deal": 0.5, "trade agreement": 0.5,
    "embargo": -0.6, "import ban": -0.5, "export ban": -0.5,
    "trade deficit": -0.3, "trade surplus": 0.2,
    # Economic crisis
    "recession": -0.7, "default": -0.8, "bankruptcy": -0.6, "bailout": -0.4,
    "inflation": -0.3, "deflation": -0.4, "stagflation": -0.6,
    "unemployment": -0.4, "layoffs": -0.5, "hiring": 0.3,
    # Policy
    "rate hike": -0.4, "rate cut": 0.4, "quantitative easing": 0.3,
    "stimulus": 0.4, "shutdown": -0.5, "debt ceiling": -0.4,
    # Market events
    "crash": -0.8, "correction": -0.4, "rally": 0.5, "surge": 0.4,
    "plunge": -0.6, "selloff": -0.5, "bull market": 0.5, "bear market": -0.5,
}

# Company name to ticker mapping (most common S&P 500)
COMPANY_TICKER_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "tesla": "TSLA", "jpmorgan": "JPM", "jp morgan": "JPM",
    "johnson & johnson": "JNJ", "j&j": "JNJ", "visa": "V",
    "procter & gamble": "PG", "p&g": "PG", "unitedhealth": "UNH",
    "home depot": "HD", "disney": "DIS", "mastercard": "MA",
    "paypal": "PYPL", "netflix": "NFLX", "adobe": "ADBE",
    "salesforce": "CRM", "intel": "INTC", "cisco": "CSCO",
    "pepsi": "PEP", "pepsico": "PEP", "coca-cola": "KO", "coke": "KO",
    "merck": "MRK", "exxon": "XOM", "exxonmobil": "XOM",
    "chevron": "CVX", "walmart": "WMT", "costco": "COST",
    "boeing": "BA", "caterpillar": "CAT", "goldman sachs": "GS",
    "morgan stanley": "MS", "bank of america": "BAC",
    "wells fargo": "WFC", "pfizer": "PFE", "eli lilly": "LLY",
    "berkshire": "BRK-B", "amd": "AMD", "qualcomm": "QCOM",
    "broadcom": "AVGO", "micron": "MU",
}


def _init_db():
    """Initialize the sentiment cache database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT,
            summary TEXT,
            url TEXT UNIQUE,
            published_at TEXT,
            ticker TEXT,
            sentiment_score REAL,
            conflict_score REAL,
            scraped_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS market_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            fear_greed_index REAL,
            vix_level REAL,
            put_call_ratio REAL,
            advance_decline REAL,
            new_highs_lows REAL,
            source TEXT,
            scraped_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS economic_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            indicator TEXT,
            value REAL,
            source TEXT,
            scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, indicator)
        );
        CREATE TABLE IF NOT EXISTS insider_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            insider_name TEXT,
            trade_type TEXT,
            shares INTEGER,
            value REAL,
            source TEXT,
            scraped_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_articles(ticker);
        CREATE INDEX IF NOT EXISTS idx_news_date ON news_articles(published_at);
        CREATE INDEX IF NOT EXISTS idx_insider_ticker ON insider_trades(ticker);
    """)
    conn.close()


def _get_db():
    """Get database connection."""
    _init_db()
    return sqlite3.connect(str(DB_PATH))


def _extract_tickers(text: str) -> list[str]:
    """Extract ticker symbols mentioned in text."""
    tickers = set()
    text_lower = text.lower()

    # Check company name mappings
    for company, ticker in COMPANY_TICKER_MAP.items():
        if company in text_lower:
            tickers.add(ticker)

    # Check for explicit ticker mentions ($AAPL or (AAPL))
    for match in re.findall(r'[\$\(]([A-Z]{1,5})[\)\s,.]', text):
        if len(match) >= 2:
            tickers.add(match)

    return list(tickers)


def _score_conflict(text: str) -> float:
    """Score text for geopolitical/conflict sentiment."""
    text_lower = text.lower()
    scores = []
    for keyword, score in CONFLICT_KEYWORDS.items():
        if keyword in text_lower:
            scores.append(score)
    return sum(scores) / max(len(scores), 1) if scores else 0.0


def scrape_rss_news() -> list[dict]:
    """Scrape financial news from RSS feeds."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    articles = []
    conn = _get_db()

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link = entry.get("link", "")
                published = entry.get("published", "")

                # Skip if already in DB
                existing = conn.execute(
                    "SELECT id FROM news_articles WHERE url=?", (link,)
                ).fetchone()
                if existing:
                    continue

                # Sentiment analysis
                text = f"{title}. {summary}"
                sentiment = analyzer.polarity_scores(text)
                conflict = _score_conflict(text)
                tickers = _extract_tickers(text)

                article = {
                    "source": source,
                    "title": title,
                    "summary": summary[:500],
                    "url": link,
                    "published_at": published,
                    "sentiment_score": sentiment["compound"],
                    "conflict_score": conflict,
                    "tickers": tickers,
                }
                articles.append(article)

                # Store per-ticker entries
                ticker_str = ",".join(tickers) if tickers else "MARKET"
                conn.execute(
                    """INSERT OR IGNORE INTO news_articles
                       (source, title, summary, url, published_at, ticker,
                        sentiment_score, conflict_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (source, title, summary[:500], link, published,
                     ticker_str, sentiment["compound"], conflict),
                )

        except Exception as e:
            logger.debug("RSS scrape error for %s: %s", source, e)

    conn.commit()
    conn.close()
    logger.info("Scraped %d new articles from RSS feeds", len(articles))
    return articles


def scrape_yfinance_news(tickers: list[str]) -> list[dict]:
    """Scrape news headlines from yfinance for specific tickers."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    articles = []
    conn = _get_db()

    for ticker in tickers:
        try:
            stock = yfinance.Ticker(ticker)
            news = stock.news if hasattr(stock, "news") else []

            for item in news[:10]:
                title = item.get("title", "")
                link = item.get("link", item.get("url", ""))
                publisher = item.get("publisher", "yfinance")
                pub_time = item.get("providerPublishTime", 0)
                published = datetime.fromtimestamp(pub_time).isoformat() if pub_time else ""

                existing = conn.execute(
                    "SELECT id FROM news_articles WHERE url=?", (link,)
                ).fetchone()
                if existing:
                    continue

                sentiment = analyzer.polarity_scores(title)
                conflict = _score_conflict(title)

                article = {
                    "source": f"yfinance_{publisher}",
                    "title": title,
                    "summary": "",
                    "url": link,
                    "published_at": published,
                    "sentiment_score": sentiment["compound"],
                    "conflict_score": conflict,
                    "tickers": [ticker],
                }
                articles.append(article)

                conn.execute(
                    """INSERT OR IGNORE INTO news_articles
                       (source, title, summary, url, published_at, ticker,
                        sentiment_score, conflict_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (f"yfinance_{publisher}", title, "", link, published,
                     ticker, sentiment["compound"], conflict),
                )

        except Exception as e:
            logger.debug("yfinance news error for %s: %s", ticker, e)

    conn.commit()
    conn.close()
    logger.info("Scraped %d yfinance articles for %d tickers", len(articles), len(tickers))
    return articles


def scrape_fear_greed() -> Optional[float]:
    """Scrape CNN Fear & Greed Index."""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            score = data.get("fear_and_greed", {}).get("score")
            if score is not None:
                conn = _get_db()
                conn.execute(
                    """INSERT OR REPLACE INTO market_sentiment
                       (date, fear_greed_index, source)
                       VALUES (?, ?, 'cnn')""",
                    (datetime.now().strftime("%Y-%m-%d"), float(score)),
                )
                conn.commit()
                conn.close()
                logger.info("Fear & Greed Index: %.1f", score)
                return float(score)
    except Exception as e:
        logger.debug("Fear & Greed scrape error: %s", e)

    # Fallback: use VIX as proxy (inverse relationship)
    try:
        vix = yfinance.Ticker("^VIX").history(period="1d")
        if not vix.empty:
            vix_val = vix["Close"].iloc[-1]
            # Rough mapping: VIX 10→90(greed), VIX 40→10(fear)
            fg_proxy = max(0, min(100, 100 - (vix_val - 10) * 3))
            logger.info("Using VIX-based Fear/Greed proxy: %.1f (VIX=%.1f)", fg_proxy, vix_val)
            return fg_proxy
    except Exception:
        pass

    return None


def scrape_fred_indicators() -> dict[str, float]:
    """Fetch key economic indicators from FRED (via public JSON endpoints)."""
    # FRED series IDs for key indicators
    series = {
        "unemployment_rate": "UNRATE",
        "cpi_yoy": "CPIAUCSL",
        "gdp_growth": "A191RL1Q225SBEA",
        "fed_rate": "DFF",
        "consumer_confidence": "UMCSENT",
        "industrial_production": "INDPRO",
        "retail_sales": "RSAFS",
        "housing_starts": "HOUST",
        "initial_claims": "ICSA",
        "m2_money": "M2SL",
    }

    results = {}
    conn = _get_db()

    for name, series_id in series.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                if len(lines) > 1:
                    last_line = lines[-1]
                    parts = last_line.split(",")
                    if len(parts) == 2 and parts[1] != ".":
                        try:
                            value = float(parts[1])
                            date = parts[0]
                            results[name] = value
                            conn.execute(
                                """INSERT OR REPLACE INTO economic_indicators
                                   (date, indicator, value, source)
                                   VALUES (?, ?, ?, 'fred')""",
                                (date, name, value),
                            )
                        except ValueError:
                            pass
        except Exception as e:
            logger.debug("FRED error for %s: %s", name, e)

    conn.commit()
    conn.close()
    logger.info("Fetched %d FRED indicators", len(results))
    return results


def scrape_sec_insider(tickers: list[str]) -> list[dict]:
    """Fetch recent insider trading from SEC EDGAR."""
    trades = []
    conn = _get_db()
    headers = {"User-Agent": "StockPredictor research@example.com"}

    for ticker in tickers[:20]:  # Rate limit: max 20
        try:
            # Get CIK for ticker
            url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={datetime.now() - timedelta(days=30):%Y-%m-%d}&forms=4"
            resp = requests.get(
                f"https://efts.sec.gov/LATEST/search-index?q={ticker}&forms=4&dateRange=custom&startdt={(datetime.now() - timedelta(days=30)):%Y-%m-%d}",
                headers=headers, timeout=10,
            )
            # SEC has strict rate limits; just log what we get
            if resp.status_code == 200:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                hits = data.get("hits", {}).get("hits", [])
                for hit in hits[:5]:
                    source = hit.get("_source", {})
                    trade = {
                        "ticker": ticker,
                        "date": source.get("file_date", ""),
                        "type": "insider_filing",
                    }
                    trades.append(trade)
            time.sleep(0.2)  # Respect rate limits
        except Exception as e:
            logger.debug("SEC EDGAR error for %s: %s", ticker, e)

    conn.close()
    logger.info("Found %d insider filings", len(trades))
    return trades


def get_aggregated_sentiment(
    ticker: str = "MARKET",
    days: int = 7,
) -> dict[str, float]:
    """Get aggregated sentiment metrics for a ticker or market."""
    conn = _get_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # News sentiment
    if ticker == "MARKET":
        rows = conn.execute(
            """SELECT sentiment_score, conflict_score FROM news_articles
               WHERE scraped_at > ? ORDER BY scraped_at DESC LIMIT 200""",
            (cutoff,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT sentiment_score, conflict_score FROM news_articles
               WHERE (ticker LIKE ? OR ticker = 'MARKET') AND scraped_at > ?
               ORDER BY scraped_at DESC LIMIT 100""",
            (f"%{ticker}%", cutoff),
        ).fetchall()

    if rows:
        sentiments = [r[0] for r in rows if r[0] is not None]
        conflicts = [r[1] for r in rows if r[1] is not None]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        avg_conflict = sum(conflicts) / len(conflicts) if conflicts else 0
        sentiment_std = (sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)) ** 0.5 if len(sentiments) > 1 else 0
    else:
        avg_sentiment = 0
        avg_conflict = 0
        sentiment_std = 0

    # Fear & Greed
    fg_row = conn.execute(
        """SELECT fear_greed_index FROM market_sentiment
           WHERE date >= ? ORDER BY date DESC LIMIT 1""",
        ((datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),),
    ).fetchone()
    fear_greed = fg_row[0] if fg_row else 50.0

    # Economic indicators
    econ_rows = conn.execute(
        """SELECT indicator, value FROM economic_indicators
           ORDER BY date DESC""",
    ).fetchall()
    econ = {r[0]: r[1] for r in econ_rows}

    conn.close()

    return {
        "news_sentiment_avg": avg_sentiment,
        "news_sentiment_std": sentiment_std,
        "news_conflict_score": avg_conflict,
        "news_article_count": len(rows) if rows else 0,
        "fear_greed_index": fear_greed,
        "unemployment_rate": econ.get("unemployment_rate", 0),
        "fed_rate": econ.get("fed_rate", 0),
        "consumer_confidence": econ.get("consumer_confidence", 0),
        "cpi_yoy": econ.get("cpi_yoy", 0),
        "initial_claims": econ.get("initial_claims", 0),
    }


def scrape_all(tickers: list[str] | None = None) -> dict:
    """Run all scrapers and return aggregated data."""
    logger.info("Running all sentiment scrapers...")

    rss_articles = scrape_rss_news()
    yf_articles = scrape_yfinance_news(tickers or []) if tickers else []
    fear_greed = scrape_fear_greed()
    fred_data = scrape_fred_indicators()

    return {
        "rss_count": len(rss_articles),
        "yfinance_count": len(yf_articles),
        "fear_greed": fear_greed,
        "fred_indicators": fred_data,
        "total_articles": len(rss_articles) + len(yf_articles),
    }
