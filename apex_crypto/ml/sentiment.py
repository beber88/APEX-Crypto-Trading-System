"""FinBERT sentiment scoring pipeline for crypto news headlines.

Processes crypto news headlines through the ProsusAI/finbert model
for financial sentiment analysis. Supports batch processing, per-symbol
filtering, and aggregated sentiment computation.
"""

from __future__ import annotations

import gc
import logging
import re
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("ml.sentiment")

# FinBERT label mapping: index -> sentiment string
_FINBERT_LABELS: list[str] = ["positive", "negative", "neutral"]

# Common crypto coin names mapped to their tickers for symbol matching
_SYMBOL_ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "BNB": ["binance", "bnb"],
    "SOL": ["solana", "sol"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "DOT": ["polkadot", "dot"],
    "AVAX": ["avalanche", "avax"],
    "MATIC": ["polygon", "matic"],
    "LINK": ["chainlink", "link"],
    "UNI": ["uniswap", "uni"],
    "ATOM": ["cosmos", "atom"],
    "LTC": ["litecoin", "ltc"],
    "NEAR": ["near", "near protocol"],
    "APT": ["aptos", "apt"],
    "ARB": ["arbitrum", "arb"],
    "OP": ["optimism"],
    "FIL": ["filecoin", "fil"],
    "SHIB": ["shiba", "shib"],
}


class SentimentPipeline:
    """FinBERT-based sentiment scoring pipeline for financial headlines.

    Lazy-loads the FinBERT model on first inference call to avoid consuming
    GPU/CPU memory at startup. Supports batch processing, per-symbol
    filtering, and aggregated sentiment computation.

    Attributes:
        model_name: HuggingFace model identifier for FinBERT.
        batch_size: Number of headlines to process per forward pass.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the sentiment pipeline with configuration.

        Args:
            config: Configuration dictionary with keys:
                - finbert_model: HuggingFace model name/path.
                  Defaults to ``"ProsusAI/finbert"``.
                - finbert_batch_size: Batch size for inference.
                  Defaults to ``16``.
        """
        self.model_name: str = config.get("finbert_model", "ProsusAI/finbert")
        self.batch_size: int = config.get("finbert_batch_size", 16)
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device: Optional[torch.device] = None

        log_with_data(
            logger,
            "info",
            "SentimentPipeline initialized",
            data={
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "model_loaded": False,
            },
        )

    def _load_model(self) -> None:
        """Load the FinBERT model and tokenizer from HuggingFace.

        Detects GPU availability and places the model on the appropriate
        device. Sets the model to eval mode for inference. The first call
        may trigger a ~400MB download from the HuggingFace Hub.

        Raises:
            RuntimeError: If the model fails to load.
        """
        if self._model is not None:
            return

        log_with_data(
            logger,
            "info",
            "Loading FinBERT model",
            data={"model_name": self.model_name},
        )

        try:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self._device)
            self._model.eval()

            log_with_data(
                logger,
                "info",
                "FinBERT model loaded successfully",
                data={
                    "device": str(self._device),
                    "model_name": self.model_name,
                },
            )
        except Exception:
            logger.error("Failed to load FinBERT model", exc_info=True)
            self._model = None
            self._tokenizer = None
            self._device = None
            raise

    def score_headlines(self, headlines: list[str]) -> list[dict[str, Any]]:
        """Score a list of headlines for financial sentiment.

        Processes headlines through FinBERT in configurable batch sizes.
        Each headline receives a sentiment label, a signed score, and
        a confidence value.

        Args:
            headlines: List of headline strings to analyse.

        Returns:
            List of result dicts, one per headline, each containing:
                - headline (str): The original headline text.
                - sentiment (str): One of ``'positive'``, ``'negative'``,
                  or ``'neutral'``.
                - score (float): Signed score in [-1, 1]. Positive values
                  indicate bullish sentiment, negative values bearish.
                - confidence (float): Softmax probability of the predicted
                  class, in [0, 1].
        """
        if not headlines:
            return []

        self._load_model()

        results: list[dict[str, Any]] = []

        for batch_start in range(0, len(headlines), self.batch_size):
            batch = headlines[batch_start : batch_start + self.batch_size]
            batch_results = self._score_batch(batch)
            results.extend(batch_results)

        log_with_data(
            logger,
            "info",
            "Scored headlines batch",
            data={
                "num_headlines": len(headlines),
                "num_results": len(results),
            },
        )

        return results

    def _score_batch(self, headlines: list[str]) -> list[dict[str, Any]]:
        """Run inference on a single batch of headlines.

        Args:
            headlines: A batch of headline strings (length <= batch_size).

        Returns:
            List of scored result dicts for the batch.
        """
        inputs = self._tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)

        results: list[dict[str, Any]] = []
        for i, headline in enumerate(headlines):
            probs = probabilities[i]
            predicted_idx = torch.argmax(probs).item()
            sentiment = _FINBERT_LABELS[predicted_idx]
            confidence = probs[predicted_idx].item()

            # Compute signed score: positive prob - negative prob
            # This yields a value in [-1, 1] where positive = bullish
            positive_prob = probs[0].item()
            negative_prob = probs[1].item()
            score = positive_prob - negative_prob

            results.append(
                {
                    "headline": headline,
                    "sentiment": sentiment,
                    "score": round(score, 4),
                    "confidence": round(confidence, 4),
                }
            )

        return results

    def score_single(self, headline: str) -> dict[str, Any]:
        """Score a single headline for financial sentiment.

        Convenience wrapper around :meth:`score_headlines` for one-off
        scoring without constructing a list.

        Args:
            headline: The headline text to score.

        Returns:
            Result dict with keys ``headline``, ``sentiment``, ``score``,
            and ``confidence``.
        """
        results = self.score_headlines([headline])
        return results[0]

    def aggregate_sentiment(
        self, scores: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate multiple headline scores into an overall sentiment.

        Computes mean and confidence-weighted scores, plus the percentage
        breakdown of bullish / bearish / neutral headlines.

        Args:
            scores: List of result dicts as returned by
                :meth:`score_headlines`.

        Returns:
            Aggregated sentiment dict containing:
                - overall_sentiment (str): Dominant sentiment label.
                - mean_score (float): Arithmetic mean of signed scores.
                - weighted_score (float): Confidence-weighted mean of
                  signed scores.
                - bullish_pct (float): Fraction of positive headlines.
                - bearish_pct (float): Fraction of negative headlines.
                - neutral_pct (float): Fraction of neutral headlines.
                - num_headlines (int): Total number of headlines scored.
        """
        if not scores:
            return {
                "overall_sentiment": "neutral",
                "mean_score": 0.0,
                "weighted_score": 0.0,
                "bullish_pct": 0.0,
                "bearish_pct": 0.0,
                "neutral_pct": 0.0,
                "num_headlines": 0,
            }

        num = len(scores)
        total_score = sum(s["score"] for s in scores)
        mean_score = total_score / num

        total_confidence = sum(s["confidence"] for s in scores)
        if total_confidence > 0:
            weighted_score = sum(
                s["score"] * s["confidence"] for s in scores
            ) / total_confidence
        else:
            weighted_score = 0.0

        bullish_count = sum(1 for s in scores if s["sentiment"] == "positive")
        bearish_count = sum(1 for s in scores if s["sentiment"] == "negative")
        neutral_count = sum(1 for s in scores if s["sentiment"] == "neutral")

        bullish_pct = bullish_count / num
        bearish_pct = bearish_count / num
        neutral_pct = neutral_count / num

        # Determine overall sentiment from the weighted score
        if weighted_score > 0.05:
            overall_sentiment = "positive"
        elif weighted_score < -0.05:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        aggregated = {
            "overall_sentiment": overall_sentiment,
            "mean_score": round(mean_score, 4),
            "weighted_score": round(weighted_score, 4),
            "bullish_pct": round(bullish_pct, 4),
            "bearish_pct": round(bearish_pct, 4),
            "neutral_pct": round(neutral_pct, 4),
            "num_headlines": num,
        }

        log_with_data(
            logger,
            "info",
            "Aggregated sentiment computed",
            data=aggregated,
        )

        return aggregated

    def score_for_symbol(
        self, headlines: list[str], symbol: str
    ) -> dict[str, Any]:
        """Filter headlines by symbol and return aggregated sentiment.

        Uses keyword matching against known coin names and ticker symbols
        to identify headlines relevant to the given trading symbol, then
        scores and aggregates them.

        Args:
            headlines: List of headline strings to filter and score.
            symbol: Trading symbol (e.g. ``"BTC"``, ``"ETH"``).

        Returns:
            Aggregated sentiment dict for headlines matching the symbol.
            See :meth:`aggregate_sentiment` for the full schema.
        """
        filtered = self._filter_headlines_for_symbol(headlines, symbol)

        log_with_data(
            logger,
            "info",
            "Filtered headlines for symbol",
            data={
                "symbol": symbol,
                "total_headlines": len(headlines),
                "matched_headlines": len(filtered),
            },
        )

        if not filtered:
            return self.aggregate_sentiment([])

        scores = self.score_headlines(filtered)
        return self.aggregate_sentiment(scores)

    @staticmethod
    def _filter_headlines_for_symbol(
        headlines: list[str], symbol: str
    ) -> list[str]:
        """Return headlines that mention the given symbol or its aliases.

        Args:
            headlines: Headlines to filter.
            symbol: Uppercase trading symbol (e.g. ``"BTC"``).

        Returns:
            Subset of headlines matching the symbol.
        """
        symbol_upper = symbol.upper().rstrip("USDT").rstrip("/")

        # Collect search keywords for this symbol
        keywords: list[str] = [symbol_upper.lower()]
        aliases = _SYMBOL_ALIASES.get(symbol_upper, [])
        keywords.extend(aliases)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_keywords: list[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        # Build a single regex pattern with word boundaries
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(kw) for kw in unique_keywords) + r")\b",
            re.IGNORECASE,
        )

        return [h for h in headlines if pattern.search(h)]

    def cleanup(self) -> None:
        """Unload the model and tokenizer to free memory.

        Moves the model off GPU (if applicable), deletes references,
        and triggers garbage collection. Safe to call even if the model
        was never loaded.
        """
        if self._model is not None:
            # Move to CPU first to release GPU memory
            if self._device is not None and self._device.type == "cuda":
                self._model.cpu()
                torch.cuda.empty_cache()

            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._device = None
        gc.collect()

        log_with_data(
            logger,
            "info",
            "SentimentPipeline cleaned up",
            data={"model_unloaded": True},
        )
