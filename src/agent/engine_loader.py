"""TradingAgents engine loader with fail-fast behavior.

This module ensures TradingAgents is available and properly configured
before the trading system starts. If TradingAgents is not found or
cannot be initialized, the system fails immediately with a clear error.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)


class TradingAgentsNotFoundError(Exception):
    """Raised when TradingAgents package cannot be found."""
    pass


class TradingAgentsInitError(Exception):
    """Raised when TradingAgents fails to initialize."""
    pass


def _find_tradingagents_path() -> Path:
    """Find the TradingAgents directory in the project.

    Returns:
        Path to the TradingAgents directory.

    Raises:
        TradingAgentsNotFoundError: If directory cannot be found.
    """
    # Look for TradingAgents relative to this file's location
    # src/agent/engine_loader.py -> project_root/TradingAgents
    project_root = Path(__file__).parent.parent.parent
    ta_path = project_root / "TradingAgents"

    if ta_path.is_dir():
        return ta_path

    # Also check if it's in the parent of project root (workspace layout)
    workspace_ta = project_root.parent / "TradingAgents"
    if workspace_ta.is_dir():
        return workspace_ta

    # Check environment variable override
    env_path = os.environ.get("TRADINGAGENTS_PATH")
    if env_path:
        env_ta_path = Path(env_path)
        if env_ta_path.is_dir():
            return env_ta_path

    raise TradingAgentsNotFoundError(
        f"TradingAgents directory not found. Searched:\n"
        f"  - {ta_path}\n"
        f"  - {workspace_ta}\n"
        f"  - TRADINGAGENTS_PATH env var (not set)\n"
        f"\nPlease ensure TradingAgents is cloned in the project root:\n"
        f"  git clone https://github.com/TauricResearch/TradingAgents.git"
    )


def _add_tradingagents_to_path(ta_path: Path) -> None:
    """Add TradingAgents to Python path if not already present."""
    ta_str = str(ta_path)
    if ta_str not in sys.path:
        sys.path.insert(0, ta_str)
        logger.info("Added TradingAgents to Python path: %s", ta_str)


def verify_tradingagents() -> bool:
    """Verify that TradingAgents is available and importable.

    Returns:
        True if TradingAgents is available.

    Raises:
        TradingAgentsNotFoundError: If TradingAgents cannot be found or imported.
    """
    ta_path = _find_tradingagents_path()
    _add_tradingagents_to_path(ta_path)

    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG
        logger.info("TradingAgents verification successful")
        return True
    except ImportError as e:
        raise TradingAgentsNotFoundError(
            f"TradingAgents found at {ta_path} but failed to import: {e}\n"
            f"Please ensure all TradingAgents dependencies are installed:\n"
            f"  pip install -r {ta_path}/requirements.txt"
        ) from e


def load_tradingagents(config: Optional[Dict[str, Any]] = None) -> "TradingAgentsGraph":
    """Load and initialize TradingAgents graph.

    This is the main entry point for obtaining a TradingAgents instance.
    It ensures the package is available, adds it to the path, and creates
    an initialized TradingAgentsGraph.

    Args:
        config: Optional configuration dict. If None, uses defaults.

    Returns:
        Initialized TradingAgentsGraph instance.

    Raises:
        TradingAgentsNotFoundError: If TradingAgents cannot be found.
        TradingAgentsInitError: If TradingAgents fails to initialize.
    """
    # First verify TradingAgents is available
    ta_path = _find_tradingagents_path()
    _add_tradingagents_to_path(ta_path)

    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG
    except ImportError as e:
        raise TradingAgentsNotFoundError(
            f"Failed to import TradingAgents: {e}"
        ) from e

    # Build config
    ta_config = DEFAULT_CONFIG.copy()
    if config:
        ta_config.update(config)

    # Ensure project_dir points to our TradingAgents installation
    ta_config["project_dir"] = str(ta_path / "tradingagents")
    ta_config["data_cache_dir"] = str(ta_path / "tradingagents" / "dataflows" / "data_cache")

    try:
        # Initialize TradingAgentsGraph
        graph = TradingAgentsGraph(
            selected_analysts=["market", "social", "news", "fundamentals"],
            debug=False,
            config=ta_config,
        )
        logger.info(
            "TradingAgents initialized: provider=%s, deep_llm=%s, quick_llm=%s",
            ta_config.get("llm_provider"),
            ta_config.get("deep_think_llm"),
            ta_config.get("quick_think_llm"),
        )
        return graph
    except Exception as e:
        raise TradingAgentsInitError(
            f"TradingAgents initialization failed: {e}\n"
            f"Config used: llm_provider={ta_config.get('llm_provider')}, "
            f"deep_think_llm={ta_config.get('deep_think_llm')}"
        ) from e


def get_tradingagents_config(
    llm_provider: str = "openai",
    deep_think_llm: str = "gpt-4o-mini",
    quick_think_llm: str = "gpt-4o-mini",
    max_debate_rounds: int = 1,
    data_vendors: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a TradingAgents config dict from parameters.

    Args:
        llm_provider: LLM provider ('openai', 'anthropic', 'google').
        deep_think_llm: Model for deep thinking tasks.
        quick_think_llm: Model for quick thinking tasks.
        max_debate_rounds: Number of debate rounds.
        data_vendors: Data vendor configuration.

    Returns:
        Config dict suitable for TradingAgentsGraph.
    """
    config = {
        "llm_provider": llm_provider,
        "deep_think_llm": deep_think_llm,
        "quick_think_llm": quick_think_llm,
        "max_debate_rounds": max_debate_rounds,
        "max_risk_discuss_rounds": max_debate_rounds,
    }

    if data_vendors:
        config["data_vendors"] = data_vendors
    else:
        config["data_vendors"] = {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",  # Use yfinance to avoid API key requirements
            "news_data": "yfinance",
        }

    return config
