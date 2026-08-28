import gc
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from ml.alternative_data import (
    AlternativeDataCollector,
    AlternativeDataEvent,
    AlternativeDataStore,
    EONET_URL,
    FEATURE_NAMES,
    GDELT_URL,
    USGS_URL,
)
from ml.features import compute_features


UTC = timezone.utc


class FakeResponse:
    def __init__(self, *, json_data=None, content=None, status_code=200):
        self._json_data = json_data
        self.content = (
            content
            if content is not None
            else json.dumps(json_data).encode("utf-8")
        )
        self.status_code = status_code

    def json(self):
        if self._json_data is None:
            raise ValueError("not JSON")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses[url]
        return response() if callable(response) else response


class AlternativeDataStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = AlternativeDataStore(
            Path(self.directory.name) / "alternative.db"
        )

    def tearDown(self):
        del self.store
        gc.collect()
        self.directory.cleanup()

    @staticmethod
    def event(**overrides):
        values = {
            "category": "news",
            "source": "test",
            "ticker": "AAPL",
            "external_id": "article-1",
            "event_time": "2026-01-05T12:00:00Z",
            "available_at": "2026-01-05T12:01:00Z",
            "collected_at": "2026-01-05T12:02:00Z",
            "sentiment": 0.5,
            "payload": {"title": "Growth beat"},
        }
        values.update(overrides)
        return AlternativeDataEvent(**values)

    def test_insert_is_idempotent(self):
        event = self.event()

        self.assertTrue(self.store.add_event(event))
        self.assertFalse(self.store.add_event(event))
        self.assertEqual(1, self.store.count_events())

    def test_available_at_is_a_hard_as_of_leakage_barrier(self):
        self.store.add_event(self.event(
            external_id="late-revision",
            event_time="2026-01-02T10:00:00Z",
            available_at="2026-01-10T16:00:00Z",
            collected_at="2026-01-10T16:01:00Z",
        ))

        before_release = self.store.get_point_in_time_features(
            "AAPL", "2026-01-10T15:59:59Z", 30
        )
        after_release = self.store.get_point_in_time_features(
            "AAPL", "2026-01-10T16:00:00Z", 30
        )

        self.assertEqual(0.0, before_release["features"]["news_count"])
        self.assertEqual(1.0, after_release["features"]["news_count"])

    def test_ticker_query_includes_global_but_not_other_tickers(self):
        self.store.add_events([
            self.event(external_id="aapl"),
            self.event(
                external_id="msft",
                ticker="MSFT",
                sentiment=-1,
            ),
            self.event(
                external_id="market",
                ticker=None,
                sentiment=0,
            ),
            self.event(
                external_id="global",
                ticker="GLOBAL",
                sentiment=1,
            ),
        ])

        result = self.store.get_point_in_time_features(
            "AAPL", "2026-01-06T00:00:00Z", 7
        )

        self.assertEqual(3.0, result["features"]["news_count"])
        self.assertEqual(0.5, result["features"]["news_sentiment_mean"])
        self.assertEqual(3, result["metadata"]["included_rows"])

    def test_features_have_fixed_numeric_schema_when_empty(self):
        result = self.store.get_point_in_time_features(
            "AAPL", datetime(2026, 1, 1, tzinfo=UTC), 30
        )

        self.assertEqual(set(FEATURE_NAMES), set(result["features"]))
        self.assertTrue(all(
            isinstance(value, float) for value in result["features"].values()
        ))
        self.assertEqual({}, result["metadata"]["sources"])

    def test_latest_snapshot_uses_only_point_in_time_eligible_row(self):
        self.store.add_events([
            self.event(
                category="fundamentals",
                source="fundamentals",
                external_id="old",
                event_time="2026-01-04T00:00:00Z",
                available_at="2026-01-04T01:00:00Z",
                payload={"market_cap": 100, "pe_ratio": 20},
                sentiment=None,
            ),
            self.event(
                category="fundamentals",
                source="fundamentals",
                external_id="future",
                event_time="2026-01-05T00:00:00Z",
                available_at="2026-01-07T00:00:00Z",
                payload={"market_cap": 999, "pe_ratio": 99},
                sentiment=None,
            ),
        ])

        result = self.store.get_point_in_time_features(
            "AAPL", "2026-01-06T00:00:00Z", 30
        )

        self.assertEqual(100.0, result["features"]["market_cap_latest"])
        self.assertEqual(20.0, result["features"]["pe_ratio_latest"])

    def test_batched_feature_frame_matches_point_in_time_queries(self):
        self.store.add_events([
            self.event(
                external_id="first",
                event_time="2026-01-05T12:00:00Z",
                available_at="2026-01-05T12:01:00Z",
                sentiment=1.0,
            ),
            self.event(
                external_id="second",
                event_time="2026-01-06T12:00:00Z",
                available_at="2026-01-08T12:00:00Z",
                sentiment=-1.0,
            ),
        ])
        dates = pd.DatetimeIndex([
            "2026-01-06T00:00:00Z",
            "2026-01-09T00:00:00Z",
        ])

        frame = self.store.get_feature_frame("AAPL", dates, 30)

        self.assertEqual(list(dates), list(frame.index))
        self.assertEqual(1.0, frame.iloc[0]["alt_news_count"])
        self.assertEqual(2.0, frame.iloc[1]["alt_news_count"])
        self.assertEqual(1.0, frame.iloc[0]["alt_news_sentiment_mean"])
        self.assertEqual(0.0, frame.iloc[1]["alt_news_sentiment_mean"])

    def test_compute_features_includes_only_available_alternative_data(self):
        dates = pd.date_range("2024-01-01", periods=400, freq="B", tz=UTC)
        positions = np.arange(len(dates))
        close = 100 + positions * 0.05 + np.sin(positions / 7)
        prices = pd.DataFrame(
            {
                "Open": close + np.cos(positions / 9) * 0.1,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000 + (positions % 17) * 10_000,
            },
            index=dates,
        )
        self.store.add_event(self.event(
            external_id="feature-pipeline",
            event_time=dates[290],
            available_at=dates[300] + pd.Timedelta(hours=12),
            collected_at=dates[301],
        ))

        with patch(
            "ml.features._alternative_data_store",
            return_value=self.store,
        ):
            features = compute_features(
                prices,
                ticker="AAPL",
                include_cross_asset=False,
            )

        self.assertIn("alt_news_count", features.columns)
        self.assertEqual(0.0, features.loc[dates[300], "alt_news_count"])
        self.assertEqual(1.0, features.loc[dates[301], "alt_news_count"])


class AlternativeDataCollectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = AlternativeDataStore(Path(self.directory.name) / "data.db")
        self.now = datetime(2026, 1, 20, 18, 0, tzinfo=UTC)

    def tearDown(self):
        del self.store
        gc.collect()
        self.directory.cleanup()

    def test_configured_rss_congressional_and_custom_feeds(self):
        news_url = "https://example.test/news.xml"
        social_url = "https://example.test/president.xml"
        congress_url = "https://example.test/congress.json"
        custom_url = "https://example.test/custom.json"
        rss = b"""<?xml version="1.0"?>
        <rss version="2.0"><channel><title>Feed</title><item>
          <guid>news-1</guid><title>AAPL profit surge</title>
          <link>https://example.test/1</link>
          <pubDate>Mon, 19 Jan 2026 15:00:00 GMT</pubDate>
        </item></channel></rss>"""
        social = b"""<?xml version="1.0"?>
        <rss version="2.0"><channel><title>Posts</title><item>
          <guid>post-1</guid><title>Policy update</title>
          <pubDate>Mon, 19 Jan 2026 16:00:00 GMT</pubDate>
        </item></channel></rss>"""
        session = FakeSession({
            news_url: FakeResponse(content=rss),
            social_url: FakeResponse(content=social),
            congress_url: FakeResponse(json_data={"trades": [{
                "id": "trade-1",
                "ticker": "AAPL",
                "transactionDate": "2026-01-10",
                "disclosureDate": "2026-01-15",
                "transactionType": "Purchase",
                "amount": "$1,001 - $15,000",
                "representative": "Example Member",
            }]}),
            custom_url: FakeResponse(json_data={"events": [{
                "id": "custom-1",
                "ticker": "AAPL",
                "category": "supplier_signal",
                "event_time": "2026-01-18T10:00:00Z",
                "available_at": "2026-01-18T12:00:00Z",
                "value": 3.5,
                "sentiment": 0.2,
                "attributes": {"region": "US"},
            }]}),
        })
        environment = {
            "ALT_DATA_NEWS_RSS_FEEDS": json.dumps({"configured_news": news_url}),
            "ALT_DATA_EXECUTIVE_RSS_FEEDS": json.dumps({
                "presidential_posts": social_url
            }),
            "ALT_DATA_CONGRESSIONAL_ENDPOINTS": json.dumps({
                "public_disclosures": congress_url
            }),
            "ALT_DATA_CUSTOM_JSON_FEEDS": json.dumps({"supplier": custom_url}),
        }

        with patch.dict(os.environ, environment, clear=False):
            collector = AlternativeDataCollector(
                self.store,
                session=session,
                timeout_seconds=7,
                now=lambda: self.now,
                enable_usgs=False,
                enable_eonet=False,
                enable_gdelt=False,
            )
            report = collector.collect_all(
                ["AAPL"],
                {"news", "executive", "congressional", "custom"},
            )

        self.assertEqual(4, report["ok"])
        self.assertEqual(0, report["failed"])
        self.assertEqual(4, report["inserted"])
        self.assertTrue(all(
            call[1]["timeout"] == 7 for call in session.calls
        ))
        features = self.store.get_point_in_time_features(
            "AAPL", self.now, 30
        )["features"]
        self.assertEqual(1.0, features["news_count"])
        self.assertEqual(1.0, features["executive_post_count"])
        self.assertEqual(1.0, features["congressional_buy_count"])
        self.assertEqual(1.0, features["custom_event_count"])
        self.assertEqual(3.5, features["custom_value_sum"])

    def test_source_failure_is_structured_and_durable(self):
        url = "https://example.test/failure"
        session = FakeSession({url: FakeResponse(status_code=503)})
        collector = AlternativeDataCollector(
            self.store,
            session=session,
            now=lambda: self.now,
            news_feeds=[("broken", url)],
            executive_feeds=[],
            congressional_endpoints=[],
            custom_json_feeds=[],
            enable_usgs=False,
            enable_eonet=False,
            enable_gdelt=False,
        )

        report = collector.collect_all([], {"news"})

        self.assertEqual(1, report["failed"])
        status = report["statuses"][0]
        self.assertEqual("error", status["status"])
        self.assertEqual("HTTPError", status["error_type"])
        metadata = self.store.get_point_in_time_features(
            "AAPL", self.now, 1
        )["metadata"]
        self.assertEqual(
            "error",
            metadata["latest_collection_status"]["news:broken"]["status"],
        )

    def test_public_disaster_and_geopolitical_sources_are_network_free(self):
        event_time_ms = int(
            datetime(2026, 1, 19, 10, 0, tzinfo=UTC).timestamp() * 1000
        )
        session = FakeSession({
            USGS_URL: FakeResponse(json_data={"features": [{
                "id": "quake-1",
                "properties": {
                    "mag": 5.2,
                    "time": event_time_ms,
                    "updated": event_time_ms + 60_000,
                    "place": "Test region",
                    "url": "https://example.test/quake",
                },
                "geometry": {"type": "Point", "coordinates": [1, 2, 3]},
            }]}),
            EONET_URL: FakeResponse(json_data={"events": [{
                "id": "storm-1",
                "title": "Test storm",
                "categories": [{"id": "severeStorms"}],
                "geometry": [{
                    "date": "2026-01-19T11:00:00Z",
                    "type": "Point",
                    "coordinates": [1, 2],
                }],
            }]}),
            GDELT_URL: FakeResponse(json_data={"articles": [{
                "url": "https://example.test/conflict",
                "title": "Conflict sanctions escalate",
                "seendate": "2026-01-19T12:00:00Z",
                "domain": "example.test",
                "language": "English",
                "sourcecountry": "US",
            }]}),
        })
        collector = AlternativeDataCollector(
            self.store,
            session=session,
            now=lambda: self.now,
            news_feeds=[],
            executive_feeds=[],
            congressional_endpoints=[],
            custom_json_feeds=[],
        )

        report = collector.collect_all(["AAPL"], {"events"})
        features = self.store.get_point_in_time_features(
            "AAPL", self.now, 7
        )["features"]

        self.assertEqual(3, report["ok"])
        self.assertEqual(2.0, features["disaster_event_count"])
        self.assertEqual(6.2, features["disaster_severity_sum"])
        self.assertEqual(1.0, features["geopolitical_event_count"])
        self.assertLess(features["geopolitical_sentiment_mean"], 0)
        self.assertTrue(all(
            call[1]["timeout"] == 15 for call in session.calls
        ))

    def test_sec_submissions_and_form4_transactions_use_acceptance_time(self):
        ticker_url = "https://www.sec.gov/files/company_tickers.json"
        submission_url = "https://data.sec.gov/submissions/CIK0000320193.json"
        filing_url = (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019326000001/form4.xml"
        )
        form4 = b"""<?xml version="1.0"?>
        <ownershipDocument>
          <reportingOwner><reportingOwnerId>
            <rptOwnerName>Example Insider</rptOwnerName>
          </reportingOwnerId></reportingOwner>
          <nonDerivativeTable><nonDerivativeTransaction>
            <transactionDate><value>2026-01-10</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
              <transactionShares><value>10</value></transactionShares>
              <transactionPricePerShare><value>150</value></transactionPricePerShare>
            </transactionAmounts>
          </nonDerivativeTransaction></nonDerivativeTable>
        </ownershipDocument>"""
        session = FakeSession({
            ticker_url: FakeResponse(json_data={
                "0": {"ticker": "AAPL", "cik_str": 320193}
            }),
            submission_url: FakeResponse(json_data={"filings": {"recent": {
                "accessionNumber": ["0000320193-26-000001"],
                "filingDate": ["2026-01-12"],
                "reportDate": ["2026-01-10"],
                "acceptanceDateTime": ["2026-01-12T16:30:00Z"],
                "form": ["4"],
                "primaryDocument": ["form4.xml"],
            }}}),
            filing_url: FakeResponse(content=form4),
        })
        with patch.dict(
            os.environ,
            {
                "ALT_DATA_SEC_USER_AGENT": "Backtester contact@example.com",
                "ALT_DATA_SEC_RATE_LIMIT_SECONDS": "0",
            },
            clear=False,
        ):
            collector = AlternativeDataCollector(
                self.store,
                session=session,
                now=lambda: self.now,
                news_feeds=[],
                executive_feeds=[],
                congressional_endpoints=[],
                custom_json_feeds=[],
                enable_usgs=False,
                enable_eonet=False,
                enable_gdelt=False,
            )
            report = collector.collect_all(["AAPL"], {"sec"})

        self.assertEqual(1, report["ok"])
        before = self.store.get_point_in_time_features(
            "AAPL", "2026-01-12T16:29:59Z", 30
        )["features"]
        after = self.store.get_point_in_time_features(
            "AAPL", "2026-01-12T16:30:00Z", 30
        )["features"]
        self.assertEqual(0.0, before["sec_form4_count"])
        self.assertEqual(1.0, after["sec_form4_count"])
        self.assertEqual(1.0, after["insider_buy_count"])
        self.assertEqual(1500.0, after["insider_net_value"])
        self.assertTrue(all(
            "contact@example.com" in call[1]["headers"]["User-Agent"]
            for call in session.calls
        ))

    def test_yfinance_snapshot_aggregates_fundamentals_earnings_and_options(self):
        earnings = pd.DataFrame(
            [{
                "EPS Estimate": 2.0,
                "Reported EPS": 2.2,
                "Surprise(%)": 10.0,
            }],
            index=pd.DatetimeIndex(["2026-01-15T16:00:00Z"]),
        )
        calls = pd.DataFrame({
            "volume": [100, 50],
            "impliedVolatility": [0.2, 0.3],
        })
        puts = pd.DataFrame({
            "volume": [75],
            "impliedVolatility": [0.4],
        })

        class FakeTicker:
            info = {
                "marketCap": 3_000_000,
                "trailingPE": 25,
                "priceToBook": 8,
                "revenueGrowth": 0.12,
            }
            options = ("2026-02-20",)

            @staticmethod
            def get_news(count=20):
                return [{
                    "id": "yf-news-1",
                    "content": {
                        "title": "AAPL growth beat",
                        "pubDate": "2026-01-18T12:00:00Z",
                        "canonicalUrl": {"url": "https://example.test/yf"},
                    },
                }]

            @staticmethod
            def get_earnings_dates(limit=12):
                return earnings

            @staticmethod
            def option_chain(expiration):
                return SimpleNamespace(calls=calls, puts=puts)

        collector = AlternativeDataCollector(
            self.store,
            now=lambda: self.now,
            yfinance_factory=lambda ticker: FakeTicker(),
            news_feeds=[],
            executive_feeds=[],
            congressional_endpoints=[],
            custom_json_feeds=[],
            enable_usgs=False,
            enable_eonet=False,
            enable_gdelt=False,
        )

        report = collector.collect_all(["AAPL"], {"yfinance"})
        features = self.store.get_point_in_time_features(
            "AAPL", self.now, 30
        )["features"]

        self.assertEqual(1, report["ok"])
        self.assertEqual(3_000_000.0, features["market_cap_latest"])
        self.assertEqual(25.0, features["pe_ratio_latest"])
        self.assertEqual(1.0, features["earnings_event_count"])
        self.assertEqual(10.0, features["earnings_surprise_mean"])
        self.assertEqual(0.5, features["options_put_call_volume_ratio_latest"])
        self.assertAlmostEqual(
            0.3, features["options_implied_volatility_mean_latest"]
        )


if __name__ == "__main__":
    unittest.main()
