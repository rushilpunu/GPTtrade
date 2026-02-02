"""TradingAgents engine adapter for the trading system.

This module provides a Policy-compatible interface to TradingAgents,
allowing it to be used as the decision engine in our trading system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date, datetime
from typing import Any, Dict, Optional

from .engine_loader import (
    TradingAgentsInitError,
    TradingAgentsNotFoundError,
    get_tradingagents_config,
    load_tradingagents,
    verify_tradingagents,
)
from .policy import Action, Decision, DecisionWithTrace, LLMTrace, Policy

logger = logging.getLogger(__name__)


class TradingAgentsPolicy(Policy):
    """Policy implementation using TradingAgents as the decision engine.

    This adapter wraps TradingAgents to provide our standard Policy interface,
    including decision tracing and compatibility with our risk gate system.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        deep_think_llm: str = "gpt-4o-mini",
        quick_think_llm: str = "gpt-4o-mini",
        max_debate_rounds: int = 1,
        data_vendors: Optional[Dict[str, str]] = None,
        lazy_init: bool = False,
    ) -> None:
        """Initialize TradingAgents policy.

        Args:
            llm_provider: LLM provider ('openai', 'anthropic', 'google').
            deep_think_llm: Model for deep thinking.
            quick_think_llm: Model for quick thinking.
            max_debate_rounds: Number of debate rounds.
            data_vendors: Data vendor configuration.
            lazy_init: If True, defer TradingAgents initialization.
        """
        self._provider = llm_provider
        self._deep_llm = deep_think_llm
        self._quick_llm = quick_think_llm
        self._max_rounds = max_debate_rounds
        self._data_vendors = data_vendors
        self._graph = None
        self._logger = logging.getLogger(__name__)

        if not lazy_init:
            self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure TradingAgents is initialized."""
        if self._graph is not None:
            return

        config = get_tradingagents_config(
            llm_provider=self._provider,
            deep_think_llm=self._deep_llm,
            quick_think_llm=self._quick_llm,
            max_debate_rounds=self._max_rounds,
            data_vendors=self._data_vendors,
        )

        self._graph = load_tradingagents(config)
        self._logger.info("TradingAgents policy initialized")

    def decide(self, features: Dict[str, Any], context: Dict[str, Any]) -> Decision:
        """Make a trading decision using TradingAgents.

        Args:
            features: Market features dict.
            context: Context dict with 'symbol' and optional 'trade_date'.

        Returns:
            Decision with action, confidence, and rationale.
        """
        result = self.decide_with_trace(features, context)
        return result.decision

    def decide_with_trace(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> DecisionWithTrace:
        """Make a trading decision with full tracing.

        Args:
            features: Market features dict.
            context: Context dict with 'symbol' and optional 'trade_date'.

        Returns:
            DecisionWithTrace with decision and LLM trace info.
        """
        self._ensure_initialized()

        trace = LLMTrace(
            llm_called=True,
            llm_provider=self._provider,
            llm_model=f"{self._deep_llm}/{self._quick_llm}",
        )

        # Extract symbol and date from context
        symbol = context.get("symbol", "UNKNOWN")
        trade_date = context.get("trade_date")
        if trade_date is None:
            trade_date = date.today().isoformat()
        elif isinstance(trade_date, (date, datetime)):
            trade_date = trade_date.isoformat()

        # Compute input hash for traceability
        try:
            payload = {"features": features, "context": context, "symbol": symbol, "date": trade_date}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            trace.llm_input_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        except Exception:
            trace.llm_input_hash = None

        start_time = time.time()

        try:
            # Call TradingAgents
            self._logger.info(
                "Calling TradingAgents: symbol=%s, date=%s", symbol, trade_date
            )
            final_state, decision_str = self._graph.propagate(symbol, trade_date)

            elapsed_ms = (time.time() - start_time) * 1000
            trace.llm_latency_ms = round(elapsed_ms, 2)

            # Parse the decision string
            action = self._parse_action(decision_str)
            trace.llm_action = action.value

            # Extract rationale from final state
            rationale = self._extract_rationale(final_state, decision_str)

            # Compute confidence based on debate consensus
            confidence = self._compute_confidence(final_state)

            # Compute output hash
            try:
                output_data = {
                    "action": action.value,
                    "confidence": confidence,
                    "decision_str": decision_str,
                }
                output_canonical = json.dumps(output_data, sort_keys=True, default=str)
                trace.llm_output_hash = hashlib.sha256(output_canonical.encode()).hexdigest()[:16]
            except Exception:
                pass

            self._logger.info(
                "TradingAgents decision: action=%s, confidence=%.2f, latency_ms=%.0f",
                action.value,
                confidence,
                trace.llm_latency_ms,
            )

            decision = Decision(
                action=action,
                confidence=confidence,
                rationale=rationale,
            )

            return DecisionWithTrace(
                decision=decision,
                trace=trace,
                policy_name="tradingagents",
            )

        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            trace.llm_latency_ms = round(elapsed_ms, 2)
            trace.used_fallback = True
            trace.llm_fallback_reason = "exception"

            self._logger.exception(
                "TradingAgents failed: %s. NO FALLBACK - raising error.", exc
            )
            # NO FALLBACK - fail fast as per requirements
            raise RuntimeError(
                f"TradingAgents decision failed for {symbol}: {exc}"
            ) from exc

    def _parse_action(self, decision_str: str) -> Action:
        """Parse TradingAgents decision string to Action enum.

        TradingAgents returns: 'BUY', 'SELL', or 'HOLD'
        We need to map to our 5-level action enum.
        """
        if decision_str is None:
            return Action.HOLD

        cleaned = str(decision_str).strip().upper()

        # Direct mapping
        if "STRONG_BUY" in cleaned or "STRONGLY BUY" in cleaned:
            return Action.STRONG_BUY
        if "STRONG_SELL" in cleaned or "STRONGLY SELL" in cleaned:
            return Action.STRONG_SELL
        if "BUY" in cleaned:
            return Action.BUY
        if "SELL" in cleaned:
            return Action.SELL
        return Action.HOLD

    def _extract_rationale(self, final_state: Dict[str, Any], decision_str: str) -> str:
        """Extract rationale from TradingAgents final state."""
        parts = ["TradingAgents decision"]

        # Try to get investment plan
        investment_plan = final_state.get("investment_plan")
        if investment_plan:
            # Truncate if too long
            if len(str(investment_plan)) > 200:
                parts.append(str(investment_plan)[:200] + "...")
            else:
                parts.append(str(investment_plan))

        # Include the raw decision
        parts.append(f"Raw: {decision_str}")

        return "; ".join(parts)

    def _compute_confidence(self, final_state: Dict[str, Any]) -> float:
        """Compute confidence score from TradingAgents state.

        Since TradingAgents doesn't directly return confidence,
        we estimate it based on debate consensus.
        """
        base_confidence = 0.7  # Default confidence

        # Check if we have debate state
        invest_debate = final_state.get("investment_debate_state", {})
        risk_debate = final_state.get("risk_debate_state", {})

        # If judge decisions exist, we have more confidence
        if invest_debate.get("judge_decision"):
            base_confidence += 0.1
        if risk_debate.get("judge_decision"):
            base_confidence += 0.1

        # Cap at 0.95
        return min(0.95, base_confidence)

    def reflect_and_remember(self, returns: float) -> None:
        """Allow TradingAgents to learn from trade outcomes.

        Args:
            returns: Position returns (positive = profit, negative = loss).
        """
        if self._graph is not None:
            try:
                self._graph.reflect_and_remember(returns)
                self._logger.info("TradingAgents reflected on returns: %.2f", returns)
            except Exception as exc:
                self._logger.warning("TradingAgents reflection failed: %s", exc)


def create_tradingagents_policy(config: Dict[str, Any]) -> TradingAgentsPolicy:
    """Factory function to create TradingAgentsPolicy from config.

    Args:
        config: Configuration dict with:
            - llm_provider: str
            - llm_model: str (used for both deep and quick think)
            - tradingagents_max_debate_rounds: int (optional)
            - tradingagents_data_vendors: dict (optional)

    Returns:
        Configured TradingAgentsPolicy instance.
    """
    provider = config.get("llm_provider", "openai")
    model = config.get("llm_model", "gpt-4o-mini")
    max_rounds = config.get("tradingagents_max_debate_rounds", 1)
    data_vendors = config.get("tradingagents_data_vendors")

    return TradingAgentsPolicy(
        llm_provider=provider,
        deep_think_llm=model,
        quick_think_llm=model,
        max_debate_rounds=max_rounds,
        data_vendors=data_vendors,
        lazy_init=False,
    )
