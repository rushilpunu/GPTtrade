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

        if trend_score > 0.05 and return_anomaly_zscore > 1.5:
            action = Action.STRONG_BUY
        elif trend_score > 0.02:
            action = Action.BUY
        elif trend_score < -0.05 and return_anomaly_zscore < -1.5:
            action = Action.STRONG_SELL
        elif trend_score < -0.02:
            action = Action.SELL
        else:
            action = Action.HOLD

        return Decision(action=action, confidence=0.6, rationale="Rules-based decision")


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
