#!/usr/bin/env python3
"""
Diagnostic audit tool for GPTtrade system.

Verifies that all components are properly wired and configured.
Run with: uv run python tools/audit.py
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class AuditResult:
    """Result of a single audit check."""
    component: str
    check: str
    status: str  # "PASS", "FAIL", "WARN"
    message: str


def check_import(module_path: str, class_name: Optional[str] = None) -> AuditResult:
    """Check if a module/class can be imported."""
    try:
        module = importlib.import_module(module_path)
        if class_name:
            if hasattr(module, class_name):
                return AuditResult(
                    component=module_path,
                    check=f"import {class_name}",
                    status="PASS",
                    message=f"{class_name} found in {module_path}",
                )
            else:
                return AuditResult(
                    component=module_path,
                    check=f"import {class_name}",
                    status="FAIL",
                    message=f"{class_name} not found in {module_path}",
                )
        return AuditResult(
            component=module_path,
            check="import module",
            status="PASS",
            message=f"Module {module_path} imported successfully",
        )
    except ImportError as e:
        return AuditResult(
            component=module_path,
            check="import module",
            status="FAIL",
            message=f"Import error: {e}",
        )


def check_config_file() -> AuditResult:
    """Check if config.yaml exists and is valid."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        return AuditResult(
            component="config",
            check="config.yaml exists",
            status="FAIL",
            message="config.yaml not found",
        )
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        required_keys = ["trading_mode", "symbols", "risk"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            return AuditResult(
                component="config",
                check="required keys",
                status="WARN",
                message=f"Missing keys: {missing}",
            )
        return AuditResult(
            component="config",
            check="config.yaml valid",
            status="PASS",
            message="Config loaded successfully",
        )
    except Exception as e:
        return AuditResult(
            component="config",
            check="config.yaml parse",
            status="FAIL",
            message=f"Parse error: {e}",
        )


def check_sentiment_integration() -> List[AuditResult]:
    """Check sentiment module integration."""
    results = []

    # Check sentiment module exists
    results.append(check_import("features.sentiment", "LexiconSentimentScorer"))
    results.append(check_import("features.sentiment", "compute_news_features"))

    # Check it produces expected output
    try:
        from features.sentiment import compute_news_features
        features = compute_news_features(["Stock surges on earnings"])
        required_keys = ["sentiment_score", "news_volume", "major_news_flag", "news_weight"]
        missing = [k for k in required_keys if k not in features]
        if missing:
            results.append(AuditResult(
                component="sentiment",
                check="output keys",
                status="FAIL",
                message=f"Missing output keys: {missing}",
            ))
        else:
            results.append(AuditResult(
                component="sentiment",
                check="output keys",
                status="PASS",
                message="All sentiment feature keys present",
            ))
    except Exception as e:
        results.append(AuditResult(
            component="sentiment",
            check="compute_news_features",
            status="FAIL",
            message=f"Error: {e}",
        ))

    return results


def check_calendar_integration() -> List[AuditResult]:
    """Check calendar module integration."""
    results = []

    results.append(check_import("data.calendar_data", "CalendarProvider"))
    results.append(check_import("data.calendar_data", "create_calendar_provider"))

    try:
        from data.calendar_data import CalendarProvider
        provider = CalendarProvider()
        flags = provider.get_calendar_flags("SPY")
        required_keys = ["earnings_soon", "fed_day", "fed_week", "blackout_active"]
        missing = [k for k in required_keys if k not in flags]
        if missing:
            results.append(AuditResult(
                component="calendar",
                check="output keys",
                status="FAIL",
                message=f"Missing output keys: {missing}",
            ))
        else:
            results.append(AuditResult(
                component="calendar",
                check="output keys",
                status="PASS",
                message="All calendar flag keys present",
            ))
    except Exception as e:
        results.append(AuditResult(
            component="calendar",
            check="CalendarProvider",
            status="FAIL",
            message=f"Error: {e}",
        ))

    return results


def check_news_integration() -> List[AuditResult]:
    """Check news data module integration."""
    results = []

    results.append(check_import("data.news_data", "NewsProvider"))
    results.append(check_import("data.news_data", "create_news_provider"))

    try:
        from data.news_data import create_news_provider
        # Should work without API key (uses RSS fallback)
        provider = create_news_provider({})
        results.append(AuditResult(
            component="news",
            check="create without API key",
            status="PASS",
            message="NewsProvider created (RSS fallback mode)",
        ))
    except Exception as e:
        results.append(AuditResult(
            component="news",
            check="create without API key",
            status="FAIL",
            message=f"Error: {e}",
        ))

    return results


def check_policy_integration() -> List[AuditResult]:
    """Check that policy uses sentiment features."""
    results = []

    try:
        from agent.policy import RulesPolicy, Action
        policy = RulesPolicy()

        # Test with sentiment features
        features = {
            "trend_score": 0.03,
            "return_anomaly_zscore": 0.5,
            "sentiment_score": 0.5,
            "news_weight": 1.2,
            "major_news_flag": 1.0,
        }
        decision = policy.decide(features, {})

        if decision.action in [Action.BUY, Action.STRONG_BUY]:
            results.append(AuditResult(
                component="policy",
                check="sentiment affects decision",
                status="PASS",
                message=f"Policy returned {decision.action.value} with sentiment",
            ))
        else:
            results.append(AuditResult(
                component="policy",
                check="sentiment affects decision",
                status="WARN",
                message=f"Policy returned {decision.action.value}, expected BUY/STRONG_BUY",
            ))

        # Check rationale mentions sentiment
        if "sentiment" in decision.rationale.lower():
            results.append(AuditResult(
                component="policy",
                check="rationale includes sentiment",
                status="PASS",
                message="Rationale mentions sentiment",
            ))
        else:
            results.append(AuditResult(
                component="policy",
                check="rationale includes sentiment",
                status="WARN",
                message="Rationale does not mention sentiment",
            ))

    except Exception as e:
        results.append(AuditResult(
            component="policy",
            check="RulesPolicy",
            status="FAIL",
            message=f"Error: {e}",
        ))

    return results


def check_risk_gate_blackout() -> List[AuditResult]:
    """Check risk gate handles calendar blackouts."""
    results = []

    try:
        from risk.risk_gate import RiskGate

        class MockBroker:
            pass

        gate = RiskGate({}, MockBroker())

        # Test blackout blocks new buys
        account_info = {"equity": 100000, "prices": {"AAPL": 150.0}}
        features = {"blackout_active": True}
        approved, reasons = gate.check_order(
            symbol="AAPL",
            side="BUY",
            qty=10,
            current_positions=[],
            account_info=account_info,
            features=features,
        )

        if "calendar_blackout" in reasons:
            results.append(AuditResult(
                component="risk_gate",
                check="blackout blocks buy",
                status="PASS",
                message="Calendar blackout correctly blocks new buys",
            ))
        else:
            results.append(AuditResult(
                component="risk_gate",
                check="blackout blocks buy",
                status="FAIL",
                message="Calendar blackout did not block buy",
            ))

    except Exception as e:
        results.append(AuditResult(
            component="risk_gate",
            check="blackout handling",
            status="FAIL",
            message=f"Error: {e}",
        ))

    return results


def check_main_wiring() -> List[AuditResult]:
    """Check main.py has all components wired."""
    results = []

    main_path = Path(__file__).parent.parent / "src" / "main.py"
    if not main_path.exists():
        results.append(AuditResult(
            component="main",
            check="file exists",
            status="FAIL",
            message="src/main.py not found",
        ))
        return results

    content = main_path.read_text()

    # Check imports
    imports_to_check = [
        ("CalendarProvider", "calendar_data"),
        ("NewsProvider", "news_data"),
        ("compute_news_features", "sentiment"),
    ]

    for symbol, module in imports_to_check:
        if symbol in content:
            results.append(AuditResult(
                component="main",
                check=f"imports {symbol}",
                status="PASS",
                message=f"{symbol} imported in main.py",
            ))
        else:
            results.append(AuditResult(
                component="main",
                check=f"imports {symbol}",
                status="FAIL",
                message=f"{symbol} not imported in main.py",
            ))

    # Check usage patterns
    if "news_provider" in content and "get_headlines" in content:
        results.append(AuditResult(
            component="main",
            check="news_provider usage",
            status="PASS",
            message="news_provider.get_headlines called",
        ))
    else:
        results.append(AuditResult(
            component="main",
            check="news_provider usage",
            status="WARN",
            message="news_provider may not be fully wired",
        ))

    if "calendar_provider" in content and "get_calendar_flags" in content:
        results.append(AuditResult(
            component="main",
            check="calendar_provider usage",
            status="PASS",
            message="calendar_provider.get_calendar_flags called",
        ))
    else:
        results.append(AuditResult(
            component="main",
            check="calendar_provider usage",
            status="WARN",
            message="calendar_provider may not be fully wired",
        ))

    return results


def run_audit() -> Tuple[List[AuditResult], int, int, int]:
    """Run all audit checks."""
    all_results: List[AuditResult] = []

    # Core imports
    all_results.append(check_import("main", "TradingSystem"))
    all_results.append(check_import("agent.policy", "RulesPolicy"))
    all_results.append(check_import("agent.policy", "LLMPolicy"))
    all_results.append(check_import("risk.risk_gate", "RiskGate"))

    # Config
    all_results.append(check_config_file())

    # Sentiment
    all_results.extend(check_sentiment_integration())

    # Calendar
    all_results.extend(check_calendar_integration())

    # News
    all_results.extend(check_news_integration())

    # Policy integration
    all_results.extend(check_policy_integration())

    # Risk gate blackout
    all_results.extend(check_risk_gate_blackout())

    # Main wiring
    all_results.extend(check_main_wiring())

    # Count results
    passes = sum(1 for r in all_results if r.status == "PASS")
    fails = sum(1 for r in all_results if r.status == "FAIL")
    warns = sum(1 for r in all_results if r.status == "WARN")

    return all_results, passes, fails, warns


def print_results(results: List[AuditResult], passes: int, fails: int, warns: int) -> None:
    """Print audit results in a formatted table."""
    print("\n" + "=" * 80)
    print("GPTtrade System Audit Report")
    print("=" * 80 + "\n")

    # Group by component
    components = {}
    for r in results:
        if r.component not in components:
            components[r.component] = []
        components[r.component].append(r)

    for component, checks in components.items():
        print(f"\n[{component}]")
        for check in checks:
            status_symbol = {"PASS": "\u2713", "FAIL": "\u2717", "WARN": "!"}[check.status]
            status_color = {"PASS": "\033[92m", "FAIL": "\033[91m", "WARN": "\033[93m"}[check.status]
            reset = "\033[0m"
            print(f"  {status_color}{status_symbol}{reset} {check.check}: {check.message}")

    print("\n" + "-" * 80)
    print(f"Summary: {passes} passed, {fails} failed, {warns} warnings")
    print("-" * 80)

    if fails > 0:
        print("\n\033[91mAUDIT FAILED\033[0m - Please fix the issues above")
    elif warns > 0:
        print("\n\033[93mAUDIT PASSED WITH WARNINGS\033[0m")
    else:
        print("\n\033[92mAUDIT PASSED\033[0m - All checks successful")


def main():
    """Main entry point."""
    results, passes, fails, warns = run_audit()
    print_results(results, passes, fails, warns)
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
