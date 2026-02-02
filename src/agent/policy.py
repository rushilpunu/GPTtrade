"""Trading policy implementations with LLM tracing."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


@dataclass
class LLMTrace:
    """Tracing information for LLM policy decisions."""
    llm_called: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_fallback_reason: Optional[str] = None  # null|no_key|http_error|timeout|invalid_json|schema_fail|exception
    llm_latency_ms: Optional[float] = None
    llm_input_hash: Optional[str] = None
    llm_output_hash: Optional[str] = None
    llm_action: Optional[str] = None  # Raw action from LLM before risk gate
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_called": self.llm_called,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_fallback_reason": self.llm_fallback_reason,
            "llm_latency_ms": self.llm_latency_ms,
            "llm_input_hash": self.llm_input_hash,
            "llm_output_hash": self.llm_output_hash,
            "llm_action": self.llm_action,
            "used_fallback": self.used_fallback,
        }


@dataclass
class DecisionWithTrace:
    """Decision bundled with LLM trace info."""
    decision: Decision
    trace: LLMTrace
    policy_name: str  # "rules" or "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.decision.action.value,
            "confidence": self.decision.confidence,
            "rationale": self.decision.rationale,
            "policy_name": self.policy_name,
            **self.trace.to_dict(),
        }


class Policy(ABC):
    @abstractmethod
    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        raise NotImplementedError

    def decide_with_trace(self, features: Dict[str, Any], context: Dict[str, Any]) -> DecisionWithTrace:
        """Decide with full tracing. Default implementation for non-LLM policies."""
        decision = self.decide(features, context)
        return DecisionWithTrace(
            decision=decision,
            trace=LLMTrace(llm_called=False),
            policy_name="rules",
        )


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

    def decide_with_trace(self, features: Dict[str, Any], context: Dict[str, Any]) -> DecisionWithTrace:
        decision = self.decide(features, context)
        return DecisionWithTrace(
            decision=decision,
            trace=LLMTrace(llm_called=False),
            policy_name="rules",
        )


class LLMPolicy(Policy):
    def __init__(
        self,
        llm_client: LLMClient,
        fallback_policy: Optional[RulesPolicy] = None,
        provider: str = "unknown",
        model: str = "unknown",
        timeout_seconds: float = 10.0,
    ) -> None:
        self._llm_client = llm_client
        self._fallback = fallback_policy or RulesPolicy()
        self._provider = provider
        self._model = model
        self._timeout = timeout_seconds
        self._logger = logging.getLogger(__name__)

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        """Legacy interface - returns just the decision."""
        result = self.decide_with_trace(features, context)
        return result.decision

    def decide_with_trace(self, features: Dict[str, Any], context: Dict[str, Any]) -> DecisionWithTrace:
        """Full decision with LLM tracing."""
        trace = LLMTrace(
            llm_provider=self._provider,
            llm_model=self._model,
        )

        # Compute input hash for traceability
        try:
            payload = {"features": features, "context": context}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            trace.llm_input_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        except Exception:
            trace.llm_input_hash = None

        start_time = time.time()
        try:
            trace.llm_called = True
            decision = self._request_decision(features, context)
            elapsed_ms = (time.time() - start_time) * 1000
            trace.llm_latency_ms = round(elapsed_ms, 2)
            trace.llm_action = decision.action.value

            # Compute output hash
            try:
                output_data = {"action": decision.action.value, "confidence": decision.confidence}
                output_canonical = json.dumps(output_data, sort_keys=True)
                trace.llm_output_hash = hashlib.sha256(output_canonical.encode()).hexdigest()[:16]
            except Exception:
                pass

            self._logger.info(
                "LLM decision: action=%s confidence=%.2f latency_ms=%.0f provider=%s",
                decision.action.value, decision.confidence, trace.llm_latency_ms, self._provider
            )

            return DecisionWithTrace(
                decision=decision,
                trace=trace,
                policy_name="llm",
            )

        except LLMClientError as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            trace.llm_latency_ms = round(elapsed_ms, 2)
            trace.used_fallback = True

            # Categorize the error
            error_str = str(exc).lower()
            if "timeout" in error_str:
                trace.llm_fallback_reason = "timeout"
            elif "http" in error_str or "api" in error_str or "request" in error_str:
                trace.llm_fallback_reason = "http_error"
            elif "json" in error_str or "parse" in error_str:
                trace.llm_fallback_reason = "invalid_json"
            elif "schema" in error_str or "validate" in error_str:
                trace.llm_fallback_reason = "schema_fail"
            else:
                trace.llm_fallback_reason = "exception"

            self._logger.warning(
                "LLM policy failed (%s), falling back to rules: %s",
                trace.llm_fallback_reason, exc
            )
            fallback_decision = self._fallback.decide(features, context)
            return DecisionWithTrace(
                decision=fallback_decision,
                trace=trace,
                policy_name="llm",  # Still "llm" policy, but with fallback flag
            )

        except (ValueError, ValidationError) as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            trace.llm_latency_ms = round(elapsed_ms, 2)
            trace.used_fallback = True
            trace.llm_fallback_reason = "schema_fail"

            self._logger.warning("LLM response validation failed, falling back: %s", exc)
            fallback_decision = self._fallback.decide(features, context)
            return DecisionWithTrace(
                decision=fallback_decision,
                trace=trace,
                policy_name="llm",
            )

        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            trace.llm_latency_ms = round(elapsed_ms, 2)
            trace.used_fallback = True
            trace.llm_fallback_reason = "exception"

            self._logger.exception("Unexpected LLM policy failure, falling back: %s", exc)
            fallback_decision = self._fallback.decide(features, context)
            return DecisionWithTrace(
                decision=fallback_decision,
                trace=trace,
                policy_name="llm",
            )

    def _request_decision(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        schema = _decision_schema()
        system_prompt = (
            "You are a trading policy model. Analyze the market features and context provided. "
            "Return a JSON decision with these exact fields:\n"
            "- action: one of STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY\n"
            "- confidence: a number between 0.0 and 1.0 indicating your confidence\n"
            "- rationale: a brief explanation of your reasoning\n"
            "Consider trend signals, sentiment, volatility, and any calendar events. "
            "Be conservative with STRONG_BUY/STRONG_SELL - use them only for clear signals."
        )
        user_payload = {"features": features, "context": context}
        data = self._llm_client.generate_json(system_prompt, user_payload, schema)
        return _validate_decision(data)


def _decision_schema() -> Dict[str, Any]:
    if hasattr(Decision, "model_json_schema"):
        return Decision.model_json_schema()
    return Decision.schema()


def _validate_decision(data: Dict[str, Any]) -> Decision:
    # Convert string action to Action enum if needed
    if "action" in data and isinstance(data["action"], str):
        try:
            data["action"] = Action(data["action"])
        except ValueError:
            pass  # Let validation catch it

    if hasattr(Decision, "model_validate"):
        return Decision.model_validate(data, strict=False)  # Allow coercion
    return Decision.parse_obj(data)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
