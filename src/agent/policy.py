from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError, confloat

from .llm_clients import LLMClient, LLMClientError

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - supports pydantic v1
    ConfigDict = None


class Action(str, Enum):
    STRONG_SELL = "STRONG_SELL"
    SELL = "SELL"
    HOLD = "HOLD"
    BUY = "BUY"
    STRONG_BUY = "STRONG_BUY"


class Decision(BaseModel):
    action: Action
    confidence: confloat(strict=True, ge=0.0, le=1.0) = Field(...)
    rationale: str

    if ConfigDict is not None:  # pydantic v2
        model_config = ConfigDict(extra="forbid")
    else:  # pydantic v1
        class Config:
            extra = "forbid"


class Policy(ABC):
    @abstractmethod
    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        raise NotImplementedError


class RulesPolicy(Policy):
    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        trend_score = _safe_float(features.get("trend_score"), default=0.0)
        return_anomaly_zscore = _safe_float(
            features.get("return_anomaly_zscore"), default=0.0
        )
        sentiment_score = _safe_float(features.get("sentiment_score"), default=0.0)
        news_weight = _safe_float(features.get("news_weight"), default=1.0)
        major_news_flag = _safe_float(features.get("major_news_flag"), default=0.0)

        # Base signal from trend
        signal_strength = trend_score

        # Amplify signal if sentiment aligns with trend
        if (trend_score > 0 and sentiment_score > 0.2) or (trend_score < 0 and sentiment_score < -0.2):
            signal_strength *= (1.0 + abs(sentiment_score) * 0.5)
        # Dampen signal if sentiment conflicts
        elif (trend_score > 0 and sentiment_score < -0.3) or (trend_score < 0 and sentiment_score > 0.3):
            signal_strength *= 0.5

        # Determine action based on adjusted signal
        if signal_strength > 0.05 and return_anomaly_zscore > 1.5:
            action = Action.STRONG_BUY
        elif signal_strength > 0.02:
            action = Action.BUY
        elif signal_strength < -0.05 and return_anomaly_zscore < -1.5:
            action = Action.STRONG_SELL
        elif signal_strength < -0.02:
            action = Action.SELL
        else:
            action = Action.HOLD

        # Adjust confidence based on news alignment
        base_confidence = 0.6
        if major_news_flag > 0:
            # Major news: higher confidence if aligned, lower if not
            if (action in (Action.BUY, Action.STRONG_BUY) and sentiment_score > 0.2) or \
               (action in (Action.SELL, Action.STRONG_SELL) and sentiment_score < -0.2):
                base_confidence = min(0.85, base_confidence + 0.15)
            elif (action in (Action.BUY, Action.STRONG_BUY) and sentiment_score < -0.2) or \
                 (action in (Action.SELL, Action.STRONG_SELL) and sentiment_score > 0.2):
                base_confidence = max(0.4, base_confidence - 0.15)

        rationale_parts = ["Rules-based decision"]
        if abs(sentiment_score) > 0.1:
            sentiment_dir = "positive" if sentiment_score > 0 else "negative"
            rationale_parts.append(f"sentiment={sentiment_dir}")
        if news_weight != 1.0:
            rationale_parts.append(f"news_weight={news_weight:.2f}")

        return Decision(
            action=action,
            confidence=round(base_confidence, 2),
            rationale="; ".join(rationale_parts),
        )


class LLMPolicy(Policy):
    def __init__(
        self,
        llm_client: LLMClient,
        fallback_policy: Optional[RulesPolicy] = None,
    ) -> None:
        self._llm_client = llm_client
        self._fallback = fallback_policy or RulesPolicy()

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        try:
            return self._request_decision(features, context)
        except (LLMClientError, ValueError, ValidationError):
            logging.exception("LLM policy failed; falling back to rules policy.")
            return self._fallback.decide(features, context)
        except Exception:
            logging.exception("Unexpected LLM policy failure; falling back to rules policy.")
            return self._fallback.decide(features, context)

    def _request_decision(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        schema = _decision_schema()
        system_prompt = (
            "You are a trading policy model. Respond with JSON only. "
            "Return a decision with fields action, confidence, rationale."
        )
        user_payload = {"features": features, "context": context}
        data = self._llm_client.generate_json(system_prompt, user_payload, schema)
        return _validate_decision(data)


def _decision_schema() -> Dict[str, Any]:
    if hasattr(Decision, "model_json_schema"):
        return Decision.model_json_schema()
    return Decision.schema()


def _validate_decision(data: Dict[str, Any]) -> Decision:
    if hasattr(Decision, "model_validate"):
        return Decision.model_validate(data, strict=True)
    return Decision.parse_obj(data)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
