#!/usr/bin/env python3
"""Inspect TradingAgents package to discover real API."""

import sys
import os

# Add parent directory to path for local import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Discover and report TradingAgents API."""
    print("=" * 60)
    print("TradingAgents API Discovery Report")
    print("=" * 60)

    # Check if TradingAgents folder exists
    ta_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TradingAgents")
    if not os.path.isdir(ta_path):
        print(f"ERROR: TradingAgents directory not found at {ta_path}")
        sys.exit(1)

    print(f"\n[OK] TradingAgents directory found: {ta_path}")

    # Add TradingAgents to path
    sys.path.insert(0, ta_path)

    # Try importing the main class
    print("\n--- Import Test ---")
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        print("[OK] Successfully imported TradingAgentsGraph")
    except ImportError as e:
        print(f"[FAIL] Cannot import TradingAgentsGraph: {e}")
        sys.exit(1)

    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        print("[OK] Successfully imported DEFAULT_CONFIG")
    except ImportError as e:
        print(f"[FAIL] Cannot import DEFAULT_CONFIG: {e}")
        sys.exit(1)

    # Inspect TradingAgentsGraph
    print("\n--- TradingAgentsGraph API ---")
    print(f"Class: {TradingAgentsGraph}")
    print(f"Module: {TradingAgentsGraph.__module__}")

    # Constructor signature
    import inspect
    sig = inspect.signature(TradingAgentsGraph.__init__)
    print(f"\nConstructor: TradingAgentsGraph{sig}")
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
        print(f"  - {param_name}: default={default}")

    # Public methods
    print("\nPublic Methods:")
    for name, method in inspect.getmembers(TradingAgentsGraph, predicate=inspect.isfunction):
        if not name.startswith("_"):
            try:
                method_sig = inspect.signature(method)
                print(f"  - {name}{method_sig}")
            except (ValueError, TypeError):
                print(f"  - {name}(...)")

    # Key method: propagate
    print("\n--- propagate() Method ---")
    prop_sig = inspect.signature(TradingAgentsGraph.propagate)
    print(f"Signature: propagate{prop_sig}")
    if TradingAgentsGraph.propagate.__doc__:
        print(f"Docstring: {TradingAgentsGraph.propagate.__doc__.strip()}")

    # Default config
    print("\n--- DEFAULT_CONFIG ---")
    for key, value in DEFAULT_CONFIG.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            # Truncate long paths
            if isinstance(value, str) and len(value) > 50:
                value = value[:47] + "..."
            print(f"  {key}: {value}")

    # Required config keys for our integration
    print("\n--- Integration Requirements ---")
    print("Required config keys:")
    print("  - llm_provider: 'openai' | 'anthropic' | 'google'")
    print("  - deep_think_llm: model name (e.g., 'gpt-4o-mini')")
    print("  - quick_think_llm: model name (e.g., 'gpt-4o-mini')")
    print("  - data_vendors: dict with 'core_stock_apis', 'technical_indicators', etc.")
    print("\nMain entrypoint:")
    print("  final_state, decision = ta.propagate(ticker, date_str)")
    print("  Returns: (full_state_dict, decision_string)")
    print("  decision_string: 'BUY' | 'SELL' | 'HOLD'")

    print("\n" + "=" * 60)
    print("API Discovery Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
