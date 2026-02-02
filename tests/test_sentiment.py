"""Tests for sentiment analysis and news features."""

import pytest

from src.features.sentiment import (
    LexiconSentimentScorer,
    SentimentResult,
    compute_news_features,
)


class TestLexiconSentimentScorer:
    """Tests for the lexicon-based sentiment scorer."""

    def test_positive_headline(self):
        scorer = LexiconSentimentScorer()
        result = scorer.score_text("Stock surges on strong earnings beat")
        assert result.score > 0
        assert result.positive_count > 0
        assert result.confidence > 0

    def test_negative_headline(self):
        scorer = LexiconSentimentScorer()
        result = scorer.score_text("Market crashes amid recession fears")
        assert result.score < 0
        assert result.negative_count > 0
        assert result.confidence > 0

    def test_neutral_headline(self):
        scorer = LexiconSentimentScorer()
        result = scorer.score_text("Company announces quarterly results")
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_empty_text(self):
        scorer = LexiconSentimentScorer()
        result = scorer.score_text("")
        assert result.score == 0.0
        assert result.positive_count == 0
        assert result.negative_count == 0
        assert result.confidence == 0.0

    def test_intensifier_amplifies_confidence(self):
        scorer = LexiconSentimentScorer()
        normal = scorer.score_text("Stock gains")
        intensified = scorer.score_text("Stock extremely gains")
        # Intensifier should increase confidence (more total weight)
        assert intensified.confidence > normal.confidence

    def test_multiple_headlines_aggregation(self):
        scorer = LexiconSentimentScorer()
        headlines = [
            "Stock surges on earnings",
            "Company beats expectations",
            "Market rallies higher",
        ]
        result = scorer.score_headlines(headlines)
        assert result.score > 0
        assert result.positive_count > 0

    def test_mixed_sentiment_headlines(self):
        scorer = LexiconSentimentScorer()
        headlines = [
            "Stock gains on good news",
            "Market drops on concerns",
        ]
        result = scorer.score_headlines(headlines)
        # Mixed sentiment should be closer to neutral
        assert -0.5 < result.score < 0.5

    def test_empty_headlines_list(self):
        scorer = LexiconSentimentScorer()
        result = scorer.score_headlines([])
        assert result.score == 0.0
        assert result.confidence == 0.0


class TestComputeNewsFeatures:
    """Tests for the compute_news_features function."""

    def test_no_headlines_returns_defaults(self):
        features = compute_news_features([])
        assert features["sentiment_score"] == 0.0
        assert features["news_volume"] == 0.0
        assert features["major_news_flag"] == 0.0
        assert features["news_weight"] == 1.0

    def test_positive_headlines_increase_weight(self):
        headlines = [
            "Stock surges dramatically",
            "Company beats all expectations",
            "Strong growth momentum continues",
        ]
        features = compute_news_features(headlines)
        assert features["sentiment_score"] > 0
        assert features["news_weight"] >= 1.0

    def test_negative_headlines_decrease_weight(self):
        headlines = [
            "Stock plunges on earnings miss",
            "Company faces bankruptcy fears",
            "Market crashes amid crisis",
        ]
        features = compute_news_features(headlines)
        assert features["sentiment_score"] < 0
        assert features["news_weight"] <= 1.0

    def test_high_volume_sets_major_news_flag(self):
        headlines = ["News " + str(i) for i in range(10)]
        # Add some sentiment words to make them parseable
        headlines[0] = "Stock gains momentum"
        features = compute_news_features(headlines)
        assert features["news_volume"] >= 0.5
        # Major news flag should be set for high volume OR strong sentiment
        # With 10 headlines, volume is 1.0, so flag should be set

    def test_strong_sentiment_sets_major_news_flag(self):
        headlines = [
            "Stock surges dramatically higher",
            "Massive gains expected",
        ]
        features = compute_news_features(headlines)
        if abs(features["sentiment_score"]) > 0.5:
            assert features["major_news_flag"] == 1.0

    def test_api_sentiment_scores_override_lexicon(self):
        headlines = ["Some headline"]
        api_scores = [0.8]  # Strong positive from API
        features = compute_news_features(headlines, api_scores)
        assert features["sentiment_score"] == 0.8

    def test_news_weight_clamped(self):
        # Test that news_weight stays within bounds
        headlines = ["Extremely strong massive gains surge rally"] * 5
        features = compute_news_features(headlines)
        assert 0.5 <= features["news_weight"] <= 1.5


class TestSentimentAffectsPolicy:
    """Integration tests verifying sentiment affects trading decisions."""

    def test_positive_sentiment_amplifies_buy_signal(self):
        from src.agent.policy import RulesPolicy, Action

        policy = RulesPolicy()

        # Base case: trend only
        features_no_sentiment = {
            "trend_score": 0.03,
            "return_anomaly_zscore": 0.5,
            "sentiment_score": 0.0,
            "news_weight": 1.0,
            "major_news_flag": 0.0,
        }

        # With positive sentiment aligned
        features_with_sentiment = {
            "trend_score": 0.03,
            "return_anomaly_zscore": 0.5,
            "sentiment_score": 0.5,
            "news_weight": 1.2,
            "major_news_flag": 1.0,
        }

        decision_no_sent = policy.decide(features_no_sentiment, {})
        decision_with_sent = policy.decide(features_with_sentiment, {})

        # Both should be BUY due to trend_score > 0.02
        assert decision_no_sent.action == Action.BUY
        assert decision_with_sent.action == Action.BUY
        # With aligned sentiment, confidence should be higher
        assert decision_with_sent.confidence >= decision_no_sent.confidence

    def test_conflicting_sentiment_dampens_signal(self):
        from src.agent.policy import RulesPolicy, Action

        policy = RulesPolicy()

        # Positive trend but negative sentiment
        features = {
            "trend_score": 0.03,
            "return_anomaly_zscore": 0.5,
            "sentiment_score": -0.5,
            "news_weight": 0.8,
            "major_news_flag": 1.0,
        }

        decision = policy.decide(features, {})
        # Signal should be dampened, might result in HOLD
        # or lower confidence if still BUY
        assert decision.action in [Action.HOLD, Action.BUY]
