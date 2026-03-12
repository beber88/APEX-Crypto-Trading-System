"""Alternative data fetcher module for the APEX crypto trading system.

Fetches and processes alternative/sentiment data from multiple sources
including Fear & Greed Index, CryptoPanic news, on-chain metrics,
and exchange-specific derivatives data (funding rates, open interest,
long/short ratios).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
import ccxt.async_support as ccxt

from apex_crypto.core.data.storage import StorageManager

logger = logging.getLogger(__name__)


class AlternativeDataManager:
    """Manages fetching, scoring, and storage of alternative/sentiment data.

    Integrates with multiple external APIs and an NLP model to provide
    a comprehensive view of market sentiment and derivatives positioning
    for the APEX crypto trading system.

    Attributes:
        config: Configuration dictionary containing API keys and settings.
        storage: StorageManager instance for Redis caching and TimescaleDB
            persistence.
    """

    # Retry configuration
    _MAX_RETRIES: int = 3
    _BASE_BACKOFF_SECONDS: float = 1.0

    # Cache TTLs (seconds)
    _FEAR_GREED_CACHE_TTL: int = 3600  # 1 hour

    # API endpoints
    _FEAR_GREED_URL: str = "https://api.alternative.me/fng/"
    _CRYPTOPANIC_URL: str = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, config: dict, storage: StorageManager) -> None:
        """Initializes the AlternativeDataManager.

        Args:
            config: Configuration dictionary. Expected keys:
                - cryptopanic_api_key: API key for CryptoPanic.
                - mexc (optional): Dict with apiKey / secret for MEXC.
                - finbert_device (optional): Torch device string, e.g. "cpu"
                  or "cuda:0". Defaults to "cpu".
            storage: StorageManager instance providing Redis cache and
                TimescaleDB access.
        """
        self.config: dict = config
        self.storage: StorageManager = storage

        self._finbert_pipeline: Any | None = None
        self._mexc_exchange: ccxt.mexc | None = None
        self._http_session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Returns a shared aiohttp session, creating one if needed.

        Returns:
            An open aiohttp.ClientSession.
        """
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def _http_get_with_retry(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> dict | list:
        """Performs an HTTP GET with exponential-backoff retries.

        Args:
            url: The URL to fetch.
            params: Optional query parameters.
            headers: Optional HTTP headers.

        Returns:
            Parsed JSON response body as a dict or list.

        Raises:
            aiohttp.ClientError: If all retry attempts are exhausted.
        """
        session = await self._get_http_session()
        last_exception: BaseException | None = None

        for attempt in range(self._MAX_RETRIES):
            try:
                async with session.get(
                    url, params=params, headers=headers
                ) as response:
                    response.raise_for_status()
                    return await response.json(content_type=None)
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
            ) as exc:
                last_exception = exc
                wait = self._BASE_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    json.dumps(
                        {
                            "event": "http_retry",
                            "url": url,
                            "attempt": attempt + 1,
                            "wait_seconds": wait,
                            "error": str(exc),
                        }
                    )
                )
                await asyncio.sleep(wait)

        raise aiohttp.ClientError(
            f"All {self._MAX_RETRIES} retries exhausted for {url}: "
            f"{last_exception}"
        )

    # ------------------------------------------------------------------
    # MEXC exchange helper
    # ------------------------------------------------------------------

    def _get_mexc_exchange(self) -> ccxt.mexc:
        """Returns a shared MEXC ccxt exchange instance.

        Returns:
            A configured ccxt.mexc async exchange object.
        """
        if self._mexc_exchange is None:
            mexc_config: dict = self.config.get("mexc", {})
            self._mexc_exchange = ccxt.mexc(
                {
                    "apiKey": mexc_config.get("apiKey", ""),
                    "secret": mexc_config.get("secret", ""),
                    "enableRateLimit": True,
                    "options": {"defaultType": "swap"},
                }
            )
        return self._mexc_exchange

    # ------------------------------------------------------------------
    # 1. Fear & Greed Index
    # ------------------------------------------------------------------

    async def fetch_fear_greed_index(self) -> dict:
        """Fetches the latest Crypto Fear & Greed Index.

        Checks Redis cache first (1-hour TTL).  On a cache miss the value
        is fetched from the alternative.me API, cached in Redis, and
        persisted to the TimescaleDB ``sentiment_scores`` table.

        Returns:
            A dict with keys:
                - value (int): Index value from 0 (extreme fear) to 100
                  (extreme greed).
                - classification (str): Human-readable classification such
                  as "Fear", "Greed", etc.
                - timestamp (datetime): UTC timestamp of the reading.
        """
        cache_key = "alt:fear_greed_index"

        try:
            cached = await self.storage.redis_get(cache_key)
            if cached is not None:
                data = json.loads(cached)
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                logger.info(
                    json.dumps(
                        {
                            "event": "fear_greed_cache_hit",
                            "value": data["value"],
                        }
                    )
                )
                return data
        except Exception as exc:
            logger.warning(
                json.dumps(
                    {
                        "event": "fear_greed_cache_read_error",
                        "error": str(exc),
                    }
                )
            )

        try:
            raw = await self._http_get_with_retry(
                self._FEAR_GREED_URL, params={"limit": 1}
            )
            entry = raw["data"][0]

            result: dict = {
                "value": int(entry["value"]),
                "classification": entry["value_classification"],
                "timestamp": datetime.fromtimestamp(
                    int(entry["timestamp"]), tz=timezone.utc
                ),
            }

            # Cache in Redis
            serializable = {
                **result,
                "timestamp": result["timestamp"].isoformat(),
            }
            try:
                await self.storage.redis_set(
                    cache_key,
                    json.dumps(serializable),
                    ttl=self._FEAR_GREED_CACHE_TTL,
                )
            except Exception as exc:
                logger.warning(
                    json.dumps(
                        {
                            "event": "fear_greed_cache_write_error",
                            "error": str(exc),
                        }
                    )
                )

            # Persist to TimescaleDB
            try:
                await self.storage.timescaledb_insert(
                    table="sentiment_scores",
                    record={
                        "source": "fear_greed_index",
                        "symbol": None,
                        "score": result["value"],
                        "classification": result["classification"],
                        "timestamp": result["timestamp"],
                    },
                )
            except Exception as exc:
                logger.warning(
                    json.dumps(
                        {
                            "event": "fear_greed_db_write_error",
                            "error": str(exc),
                        }
                    )
                )

            logger.info(
                json.dumps(
                    {
                        "event": "fear_greed_fetched",
                        "value": result["value"],
                        "classification": result["classification"],
                    }
                )
            )
            return result

        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "fear_greed_fetch_error",
                        "error": str(exc),
                    }
                )
            )
            return {
                "value": -1,
                "classification": "unavailable",
                "timestamp": datetime.now(tz=timezone.utc),
            }

    # ------------------------------------------------------------------
    # 2. Crypto News
    # ------------------------------------------------------------------

    async def fetch_crypto_news(
        self, symbol: str | None = None
    ) -> list[dict]:
        """Fetches recent crypto news headlines from CryptoPanic.

        Args:
            symbol: Optional currency ticker (e.g. "BTC") to filter news.
                When ``None``, returns news for all currencies.

        Returns:
            A list of dicts, each containing:
                - title (str): News headline.
                - source (str): Source publication name.
                - url (str): Link to the article.
                - published_at (str): ISO-8601 publication timestamp.
                - currencies (list[str]): Tickers of related currencies.
                - kind (str): Article type, e.g. "news", "media".
        """
        api_key: str = self.config.get("cryptopanic_api_key", "")
        if not api_key:
            logger.error(
                json.dumps(
                    {
                        "event": "cryptopanic_missing_api_key",
                        "error": "cryptopanic_api_key not set in config",
                    }
                )
            )
            return []

        params: dict[str, str] = {
            "auth_token": api_key,
            "public": "true",
        }
        if symbol is not None:
            params["currencies"] = symbol.upper()

        try:
            raw = await self._http_get_with_retry(
                self._CRYPTOPANIC_URL, params=params
            )
            results: list[dict] = []
            for post in raw.get("results", []):
                currencies_raw = post.get("currencies") or []
                results.append(
                    {
                        "title": post.get("title", ""),
                        "source": (post.get("source") or {}).get("title", ""),
                        "url": post.get("url", ""),
                        "published_at": post.get("published_at", ""),
                        "currencies": [
                            c.get("code", "") for c in currencies_raw
                        ],
                        "kind": post.get("kind", ""),
                    }
                )

            logger.info(
                json.dumps(
                    {
                        "event": "cryptopanic_fetched",
                        "symbol": symbol,
                        "count": len(results),
                    }
                )
            )
            return results

        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "cryptopanic_fetch_error",
                        "symbol": symbol,
                        "error": str(exc),
                    }
                )
            )
            return []

    # ------------------------------------------------------------------
    # 3. FinBERT Sentiment Scoring
    # ------------------------------------------------------------------

    def _load_finbert(self) -> Any:
        """Lazy-loads the ProsusAI/finbert sentiment-analysis pipeline.

        The pipeline is loaded once and retained in memory for subsequent
        calls.

        Returns:
            A HuggingFace transformers sentiment-analysis pipeline.
        """
        if self._finbert_pipeline is None:
            logger.info(
                json.dumps({"event": "finbert_loading", "model": "ProsusAI/finbert"})
            )
            from transformers import pipeline as hf_pipeline

            device_str: str = self.config.get("finbert_device", "cpu")
            device: int = -1  # CPU default
            if device_str.startswith("cuda"):
                parts = device_str.split(":")
                device = int(parts[1]) if len(parts) > 1 else 0

            self._finbert_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=device,
            )
            logger.info(json.dumps({"event": "finbert_loaded"}))
        return self._finbert_pipeline

    async def score_sentiment_finbert(
        self, headlines: list[str]
    ) -> list[dict]:
        """Scores a list of headlines using the ProsusAI/finbert model.

        The model is lazy-loaded on the first invocation and kept in
        memory for subsequent calls.  Inference is offloaded to a thread
        pool so the event loop is not blocked.

        Args:
            headlines: A list of headline strings to score.

        Returns:
            A list of dicts, one per headline, each containing:
                - headline (str): The original headline text.
                - sentiment (str): One of "positive", "negative", or
                  "neutral".
                - score (float): Sentiment score in the range -1 to 1.
                - confidence (float): Model confidence for the predicted
                  label.
        """
        if not headlines:
            return []

        try:
            pipe = self._load_finbert()

            # Run inference in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            raw_results: list[dict] = await loop.run_in_executor(
                None, pipe, headlines
            )

            scored: list[dict] = []
            for headline, result in zip(headlines, raw_results):
                label: str = result["label"].lower()
                confidence: float = float(result["score"])

                # Map label to a continuous score in [-1, 1]
                if label == "positive":
                    score = confidence
                elif label == "negative":
                    score = -confidence
                else:
                    score = 0.0

                scored.append(
                    {
                        "headline": headline,
                        "sentiment": label,
                        "score": round(score, 4),
                        "confidence": round(confidence, 4),
                    }
                )

            logger.info(
                json.dumps(
                    {
                        "event": "finbert_scored",
                        "count": len(scored),
                    }
                )
            )
            return scored

        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "finbert_scoring_error",
                        "error": str(exc),
                    }
                )
            )
            return [
                {
                    "headline": h,
                    "sentiment": "unknown",
                    "score": 0.0,
                    "confidence": 0.0,
                }
                for h in headlines
            ]

    # ------------------------------------------------------------------
    # 4. Funding Rates
    # ------------------------------------------------------------------

    async def fetch_funding_rates(
        self, symbols: list[str]
    ) -> dict[str, float]:
        """Fetches current perpetual funding rates from MEXC futures.

        Args:
            symbols: List of unified symbol strings, e.g.
                ``["BTC/USDT:USDT", "ETH/USDT:USDT"]``.

        Returns:
            A dict mapping each symbol to its current funding rate as a
            float.  Symbols that could not be fetched are omitted.
        """
        exchange = self._get_mexc_exchange()
        rates: dict[str, float] = {}

        try:
            await exchange.load_markets()

            for symbol in symbols:
                try:
                    funding = await exchange.fetch_funding_rate(symbol)
                    rate: float = float(funding.get("fundingRate", 0.0))
                    rates[symbol] = rate

                    # Persist to TimescaleDB
                    try:
                        await self.storage.timescaledb_insert(
                            table="funding_rates",
                            record={
                                "symbol": symbol,
                                "rate": rate,
                                "timestamp": datetime.now(tz=timezone.utc),
                            },
                        )
                    except Exception as db_exc:
                        logger.warning(
                            json.dumps(
                                {
                                    "event": "funding_rate_db_write_error",
                                    "symbol": symbol,
                                    "error": str(db_exc),
                                }
                            )
                        )

                except Exception as exc:
                    logger.warning(
                        json.dumps(
                            {
                                "event": "funding_rate_fetch_error",
                                "symbol": symbol,
                                "error": str(exc),
                            }
                        )
                    )

            logger.info(
                json.dumps(
                    {
                        "event": "funding_rates_fetched",
                        "count": len(rates),
                    }
                )
            )

        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "funding_rates_load_markets_error",
                        "error": str(exc),
                    }
                )
            )
        finally:
            await exchange.close()
            self._mexc_exchange = None

        return rates

    # ------------------------------------------------------------------
    # 5. Funding Rate Average
    # ------------------------------------------------------------------

    async def compute_funding_rate_average(
        self, symbol: str, days: int = 7
    ) -> float:
        """Computes the average funding rate over a configurable window.

        Queries the TimescaleDB ``funding_rates`` table for historical
        records.

        Args:
            symbol: Unified symbol string, e.g. "BTC/USDT:USDT".
            days: Number of days to look back. Defaults to 7.

        Returns:
            The mean funding rate over the period, or 0.0 if no data is
            available.
        """
        try:
            query = (
                "SELECT AVG(rate) AS avg_rate "
                "FROM funding_rates "
                "WHERE symbol = $1 "
                "  AND timestamp >= NOW() - make_interval(days => $2)"
            )
            rows = await self.storage.timescaledb_query(
                query=query, params=[symbol, days]
            )

            if rows and rows[0].get("avg_rate") is not None:
                avg: float = float(rows[0]["avg_rate"])
                logger.info(
                    json.dumps(
                        {
                            "event": "funding_rate_avg_computed",
                            "symbol": symbol,
                            "days": days,
                            "average": avg,
                        }
                    )
                )
                return avg

            logger.info(
                json.dumps(
                    {
                        "event": "funding_rate_avg_no_data",
                        "symbol": symbol,
                        "days": days,
                    }
                )
            )
            return 0.0

        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "funding_rate_avg_error",
                        "symbol": symbol,
                        "error": str(exc),
                    }
                )
            )
            return 0.0

    # ------------------------------------------------------------------
    # 6. Open Interest
    # ------------------------------------------------------------------

    async def fetch_open_interest(
        self, symbols: list[str]
    ) -> dict[str, dict]:
        """Fetches open interest data from MEXC futures for given symbols.

        Args:
            symbols: List of unified symbol strings, e.g.
                ``["BTC/USDT:USDT", "ETH/USDT:USDT"]``.

        Returns:
            A dict mapping each symbol to a dict with:
                - oi_value (float): Current open interest notional value.
                - oi_change_pct_24h (float): 24-hour percentage change in
                  open interest.
            Symbols that could not be fetched are omitted.
        """
        exchange = self._get_mexc_exchange()
        oi_data: dict[str, dict] = {}

        try:
            await exchange.load_markets()

            for symbol in symbols:
                try:
                    oi = await exchange.fetch_open_interest(symbol)
                    oi_value: float = float(
                        oi.get("openInterestAmount", 0.0)
                    )

                    # Try to compute 24h change from stored data
                    oi_change_pct: float = 0.0
                    try:
                        query = (
                            "SELECT oi_value "
                            "FROM open_interest "
                            "WHERE symbol = $1 "
                            "  AND timestamp >= NOW() - INTERVAL '24 hours' "
                            "ORDER BY timestamp ASC "
                            "LIMIT 1"
                        )
                        rows = await self.storage.timescaledb_query(
                            query=query, params=[symbol]
                        )
                        if rows and rows[0].get("oi_value"):
                            old_oi = float(rows[0]["oi_value"])
                            if old_oi > 0:
                                oi_change_pct = (
                                    (oi_value - old_oi) / old_oi
                                ) * 100.0
                    except Exception:
                        pass  # Gracefully degrade; use 0.0

                    oi_data[symbol] = {
                        "oi_value": oi_value,
                        "oi_change_pct_24h": round(oi_change_pct, 4),
                    }

                    # Persist to TimescaleDB
                    try:
                        await self.storage.timescaledb_insert(
                            table="open_interest",
                            record={
                                "symbol": symbol,
                                "oi_value": oi_value,
                                "timestamp": datetime.now(tz=timezone.utc),
                            },
                        )
                    except Exception as db_exc:
                        logger.warning(
                            json.dumps(
                                {
                                    "event": "open_interest_db_write_error",
                                    "symbol": symbol,
                                    "error": str(db_exc),
                                }
                            )
                        )

                except Exception as exc:
                    logger.warning(
                        json.dumps(
                            {
                                "event": "open_interest_fetch_error",
                                "symbol": symbol,
                                "error": str(exc),
                            }
                        )
                    )

            logger.info(
                json.dumps(
                    {
                        "event": "open_interest_fetched",
                        "count": len(oi_data),
                    }
                )
            )

        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "open_interest_load_markets_error",
                        "error": str(exc),
                    }
                )
            )
        finally:
            await exchange.close()
            self._mexc_exchange = None

        return oi_data

    # ------------------------------------------------------------------
    # 7. Long/Short Ratio
    # ------------------------------------------------------------------

    async def fetch_long_short_ratio(self, symbol: str) -> dict:
        """Fetches the long/short account ratio from MEXC futures.

        Args:
            symbol: Unified symbol string, e.g. "BTC/USDT:USDT".

        Returns:
            A dict containing:
                - long_pct (float): Percentage of accounts long.
                - short_pct (float): Percentage of accounts short.
                - ratio (float): Long/short ratio.
                - timestamp (datetime): UTC timestamp of the reading.
        """
        exchange = self._get_mexc_exchange()

        try:
            await exchange.load_markets()

            market = exchange.market(symbol)
            mexc_symbol: str = market.get("id", symbol)

            response = await exchange.publicGetContractV1TradeLongShort(
                {"symbol": mexc_symbol}
            )

            data = response.get("data", {})
            long_pct: float = float(data.get("longAccountRatio", 0.5)) * 100
            short_pct: float = float(data.get("shortAccountRatio", 0.5)) * 100
            ratio: float = (
                long_pct / short_pct if short_pct > 0 else float("inf")
            )

            result: dict = {
                "long_pct": round(long_pct, 2),
                "short_pct": round(short_pct, 2),
                "ratio": round(ratio, 4),
                "timestamp": datetime.now(tz=timezone.utc),
            }

            logger.info(
                json.dumps(
                    {
                        "event": "long_short_ratio_fetched",
                        "symbol": symbol,
                        "ratio": result["ratio"],
                    }
                )
            )
            return result

        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "long_short_ratio_error",
                        "symbol": symbol,
                        "error": str(exc),
                    }
                )
            )
            now = datetime.now(tz=timezone.utc)
            return {
                "long_pct": 50.0,
                "short_pct": 50.0,
                "ratio": 1.0,
                "timestamp": now,
            }
        finally:
            await exchange.close()
            self._mexc_exchange = None

    # ------------------------------------------------------------------
    # 8. Refresh All
    # ------------------------------------------------------------------

    async def refresh_all(self, symbols: list[str]) -> dict:
        """Fetches all alternative data sources in parallel.

        Designed to be called every 15 minutes by the system scheduler.
        Individual source failures are caught and logged so that the
        remaining sources still return data (graceful degradation).

        Args:
            symbols: List of unified symbol strings to fetch data for.

        Returns:
            A combined dict with keys:
                - fear_greed (dict): Fear & Greed Index result.
                - news (list[dict]): Latest crypto news.
                - news_sentiment (list[dict]): FinBERT scores for headlines.
                - funding_rates (dict[str, float]): Current funding rates.
                - open_interest (dict[str, dict]): Open interest data.
                - long_short_ratios (dict[str, dict]): Long/short ratio per
                  symbol.
        """
        start_time: float = time.monotonic()

        # Launch independent fetches concurrently
        fear_greed_task = asyncio.ensure_future(
            self._safe_fetch("fear_greed", self.fetch_fear_greed_index)
        )
        news_task = asyncio.ensure_future(
            self._safe_fetch("news", self.fetch_crypto_news)
        )
        funding_task = asyncio.ensure_future(
            self._safe_fetch(
                "funding_rates", self.fetch_funding_rates, symbols
            )
        )
        oi_task = asyncio.ensure_future(
            self._safe_fetch(
                "open_interest", self.fetch_open_interest, symbols
            )
        )
        ls_tasks = {
            sym: asyncio.ensure_future(
                self._safe_fetch(
                    f"long_short_{sym}",
                    self.fetch_long_short_ratio,
                    sym,
                )
            )
            for sym in symbols
        }

        # Await all tasks
        all_tasks = [fear_greed_task, news_task, funding_task, oi_task] + list(
            ls_tasks.values()
        )
        await asyncio.gather(*all_tasks)

        fear_greed: dict = fear_greed_task.result()
        news: list[dict] = news_task.result()
        funding_rates: dict[str, float] = funding_task.result()
        open_interest: dict[str, dict] = oi_task.result()
        long_short_ratios: dict[str, dict] = {
            sym: task.result() for sym, task in ls_tasks.items()
        }

        # Score news headlines with FinBERT
        headlines: list[str] = [
            article["title"] for article in news if article.get("title")
        ]
        news_sentiment: list[dict] = await self._safe_fetch(
            "news_sentiment",
            self.score_sentiment_finbert,
            headlines,
        )

        elapsed: float = round(time.monotonic() - start_time, 2)
        logger.info(
            json.dumps(
                {
                    "event": "refresh_all_complete",
                    "symbols_count": len(symbols),
                    "elapsed_seconds": elapsed,
                }
            )
        )

        return {
            "fear_greed": fear_greed,
            "news": news,
            "news_sentiment": news_sentiment,
            "funding_rates": funding_rates,
            "open_interest": open_interest,
            "long_short_ratios": long_short_ratios,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _safe_fetch(
        self, name: str, coro_func: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Wraps an async call with error handling for graceful degradation.

        Args:
            name: Human-readable name of the data source (for logging).
            coro_func: The async callable to invoke.
            *args: Positional arguments forwarded to ``coro_func``.
            **kwargs: Keyword arguments forwarded to ``coro_func``.

        Returns:
            The result of ``coro_func``, or a sensible empty default on
            failure (empty dict, empty list, or ``None``).
        """
        try:
            return await coro_func(*args, **kwargs)
        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "safe_fetch_failure",
                        "source": name,
                        "error": str(exc),
                    }
                )
            )
            # Return an appropriate empty default based on type hints
            return {} if "ratio" in name or "rate" in name or "interest" in name else []

    async def close(self) -> None:
        """Releases all held resources (HTTP session, exchange connection).

        Should be called during application shutdown.
        """
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

        if self._mexc_exchange is not None:
            try:
                await self._mexc_exchange.close()
            except Exception:
                pass
            self._mexc_exchange = None

        logger.info(
            json.dumps({"event": "alternative_data_manager_closed"})
        )
