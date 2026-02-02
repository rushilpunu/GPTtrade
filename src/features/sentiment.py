"""Deterministic sentiment analysis for news headlines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Lexicon-based sentiment scoring - deterministic, no network required
_POSITIVE_WORDS = frozenset([
    "gain", "gains", "rise", "rises", "rising", "surge", "surges", "surging",
    "jump", "jumps", "jumping", "rally", "rallies", "rallying", "soar", "soars",
    "boost", "boosted", "upgrade", "upgraded", "beat", "beats", "beating",
    "outperform", "outperforms", "bullish", "optimistic", "positive", "strong",
    "stronger", "growth", "profit", "profits", "profitable", "record", "high",
    "higher", "up", "upside", "buy", "buying", "bought", "advance", "advances",
    "recover", "recovers", "recovery", "improve", "improves", "improved",
    "breakthrough", "success", "successful", "win", "wins", "winning", "boom",
    "expand", "expands", "expansion", "opportunity", "opportunities", "momentum",
])

_NEGATIVE_WORDS = frozenset([
    "loss", "losses", "lose", "loses", "losing", "fall", "falls", "falling",
    "drop", "drops", "dropping", "decline", "declines", "declining", "sink",
    "sinks", "sinking", "plunge", "plunges", "plunging", "crash", "crashes",
    "tumble", "tumbles", "downgrade", "downgraded", "miss", "misses", "missing",
    "underperform", "underperforms", "bearish", "pessimistic", "negative", "weak",
    "weaker", "slowdown", "recession", "deficit", "deficits", "low", "lower",
    "down", "downside", "sell", "selling", "sold", "selloff", "retreat", "retreats",
    "fail", "fails", "failed", "failure", "warning", "warnings", "risk", "risks",
    "concern", "concerns", "worried", "worry", "fear", "fears", "crisis", "trouble",
    "troubled", "cut", "cuts", "cutting", "layoff", "layoffs", "lawsuit", "fine",
    "penalty", "fraud", "scandal", "bankruptcy", "default", "debt", "inflation",
])

_INTENSIFIERS = frozenset([
    "very", "extremely", "highly", "significantly", "substantially", "sharply",
    "dramatically", "massively", "hugely", "strongly", "major", "big", "huge",
])


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    score: float  # -1.0 to 1.0
    positive_count: int
    negative_count: int
    confidence: float  # 0.0 to 1.0 based on word matches


class LexiconSentimentScorer:
    """Deterministic lexicon-based sentiment scorer."""

    def __init__(
        self,
        positive_words: Optional[frozenset] = None,
        negative_words: Optional[frozenset] = None,
        intensifiers: Optional[frozenset] = None,
    ) -> None:
        self._positive = positive_words or _POSITIVE_WORDS
        self._negative = negative_words or _NEGATIVE_WORDS
        self._intensifiers = intensifiers or _INTENSIFIERS

    def score_text(self, text: str) -> SentimentResult:
        """Score a single text string."""
        if not text:
            return SentimentResult(score=0.0, positive_count=0, negative_count=0, confidence=0.0)

        words = self._tokenize(text)
        positive_count = 0
        negative_count = 0
        intensifier_active = False

        for word in words:
            if word in self._intensifiers:
                intensifier_active = True
                continue

            weight = 1.5 if intensifier_active else 1.0
            intensifier_active = False

            if word in self._positive:
                positive_count += weight
            elif word in self._negative:
                negative_count += weight

        total = positive_count + negative_count
        if total == 0:
            return SentimentResult(score=0.0, positive_count=0, negative_count=0, confidence=0.0)

        raw_score = (positive_count - negative_count) / total
        confidence = min(1.0, total / 5.0)  # More words = higher confidence

        return SentimentResult(
            score=float(raw_score),
            positive_count=int(positive_count),
            negative_count=int(negative_count),
            confidence=float(confidence),
        )

    def score_headlines(self, headlines: Sequence[str]) -> SentimentResult:
        """Score multiple headlines and return aggregate."""
        if not headlines:
            return SentimentResult(score=0.0, positive_count=0, negative_count=0, confidence=0.0)

        total_positive = 0
        total_negative = 0
        weighted_scores = []

        for headline in headlines:
            result = self.score_text(headline)
            total_positive += result.positive_count
            total_negative += result.negative_count
            if result.confidence > 0:
                weighted_scores.append((result.score, result.confidence))

        if not weighted_scores:
            return SentimentResult(
                score=0.0,
                positive_count=int(total_positive),
                negative_count=int(total_negative),
                confidence=0.0,
            )

        total_weight = sum(w for _, w in weighted_scores)
        avg_score = sum(s * w for s, w in weighted_scores) / total_weight if total_weight > 0 else 0.0
        avg_confidence = total_weight / len(weighted_scores)

        return SentimentResult(
            score=float(avg_score),
            positive_count=int(total_positive),
            negative_count=int(total_negative),
            confidence=float(avg_confidence),
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize and normalize text."""
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        return [w.strip() for w in cleaned.split() if w.strip()]


def compute_news_features(
    headlines: Sequence[str],
    api_sentiment_scores: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, float]:
    """
    Compute news-based features from headlines.

    Returns:
        sentiment_score: -1.0 to 1.0 aggregate sentiment
        news_volume: number of headlines (normalized 0-1)
        major_news_flag: 1.0 if high volume or strong sentiment, else 0.0
        news_weight: 0.0 to 1.0 multiplier for position sizing
    """
    if not headlines:
        return {
            "sentiment_score": 0.0,
            "news_volume": 0.0,
            "major_news_flag": 0.0,
            "news_weight": 1.0,  # No news = neutral weight
        }

    # Use API sentiment if available, else lexicon
    if api_sentiment_scores and any(s is not None for s in api_sentiment_scores):
        valid_scores = [s for s in api_sentiment_scores if s is not None]
        sentiment_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        confidence = 0.8  # API scores assumed reliable
    else:
        scorer = LexiconSentimentScorer()
        result = scorer.score_headlines(list(headlines))
        sentiment_score = result.score
        confidence = result.confidence

    news_volume = min(1.0, len(headlines) / 10.0)  # Normalize to 0-1
    major_news_flag = 1.0 if (news_volume > 0.5 or abs(sentiment_score) > 0.5) else 0.0

    # News weight: amplify or dampen based on sentiment strength
    # Positive news = slightly higher weight, negative = lower
    if sentiment_score > 0.3:
        news_weight = 1.0 + (sentiment_score * 0.3 * confidence)
    elif sentiment_score < -0.3:
        news_weight = 1.0 + (sentiment_score * 0.5 * confidence)  # More cautious on negative
    else:
        news_weight = 1.0

    news_weight = max(0.5, min(1.5, news_weight))  # Clamp to reasonable range

    return {
        "sentiment_score": float(sentiment_score),
        "news_volume": float(news_volume),
        "major_news_flag": float(major_news_flag),
        "news_weight": float(news_weight),
    }
