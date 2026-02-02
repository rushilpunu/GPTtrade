# GPTtrade

Event-driven algorithmic trading system that turns market data into decisions and orders using behavioral signals, configurable risk controls, and pluggable policies (rules-based or LLM-driven). The system can run in a single cycle for ad‑hoc evaluation or as a scheduler loop for continuous operation.

## Features
- Behavioral signals from price/volume history
- Risk gate with position sizing and exposure limits
- Paper and live trading support (Alpaca + simulator broker)
- Policy layer: rules-based or LLM-backed decisions
- Order execution with optional limit orders and cooldowns
- Decision + order logging to a local SQLite database

## Project Structure
```
.
├── README.md
├── config.yaml
├── .env.example
├── pyproject.toml
├── src
│   ├── main.py
│   ├── scheduler.py
│   ├── agent
│   │   └── policy.py
│   ├── data
│   │   └── market_data.py
│   ├── execution
│   │   ├── alpaca_broker.py
│   │   ├── broker_interface.py
│   │   └── simulator_broker.py
│   ├── features
│   │   └── behavioral_features.py
│   ├── observability
│   │   └── logging.py
│   ├── risk
│   │   ├── position_sizing.py
│   │   └── risk_gate.py
│   └── storage
│       ├── db.py
│       └── models.py
├── tests
│   ├── test_position_sizing.py
│   └── test_risk_gate.py
└── TradingAgents
    └── ...
```

## Prerequisites
- Python 3.11+
- `uv` or `pip`

## Setup
```bash
git clone <your-repo-url>
cd GPTtrade
```

Install dependencies:
```bash
uv sync
# or
pip install -e .
```

Copy environment template:
```bash
cp .env.example .env
```

## Configuration Guide (`config.yaml`)
Key knobs you can tune:
- `trading_mode`: `paper` or `live` (execution mode selector).
- `ENABLE_LIVE_TRADING`: hard safety switch for live orders.
- `symbols`: list of tickers to trade.
- `decision_interval_minutes`: cadence for scheduler loop.
- `lookback_days`: (optional) OHLCV history window used for features.
- `max_position_pct`: max equity per position (fraction).
- `max_gross_exposure_pct`: cap on total exposure (fraction).
- `max_daily_loss_pct`: daily loss stop (fraction).
- `max_trades_per_day`: throttle for order count.
- `cooldown_minutes`: minimum wait between trades per symbol.
- `cost_buffer`: slippage/fee buffer used in sizing/validation.
- `policy_type`: `rules` or `llm`.
- `llm.model`, `llm.endpoint`, `llm.temperature`: LLM policy settings.
- `news_enabled`: include news features when enabled.
- `use_limit_orders`: toggle market vs limit orders.
- `limit_order_offset`: percent offset from mid/last for limit orders.
- `stale_order_timeout_seconds`: cancel orders past this age.
- `enable_shorts`, `enable_leverage`: additional risk toggles.
- `logging.*`: log level, outputs, rotation settings.

Environment variables in `.env` can override sensitive items (API keys, live-trading token, etc.).

## Run
Single cycle:
```bash
python -m main --once
```

Scheduler loop:
```bash
python -m main
```

Dry run (no orders submitted):
```bash
python -m main --dry-run
```

## Testing & Verification

### Run All Tests
```bash
cd src && PYTHONPATH=. pytest ../tests -v
```

### Test Categories

| Test File | Description |
|-----------|-------------|
| `test_integration.py` | Full cycle integration with rules/LLM policies, SQLite persistence, correlation IDs |
| `test_scheduler_autonomy.py` | Scheduler iteration control, graceful shutdown, dry-run mode |
| `test_no_prompting.py` | Enforces no `input()` calls, static source scan for interactive patterns |
| `test_offline.py` | Runs with network blocked, verifies LLM fallback works offline |
| `test_risk_gate.py` | Risk constraint unit tests (PDT, position limits, kill switch) |
| `test_position_sizing.py` | Signal-based sizing, short blocking, max position constraints |

### Smoke Test
Quick verification that the system runs end-to-end in simulator mode:
```bash
cd src && python ../tools/smoke.py
# Verbose mode:
cd src && python ../tools/smoke.py --verbose
```

### Continuous Integration
GitHub Actions workflow runs on push/PR:
- Installs dependencies
- Runs full pytest suite
- Runs smoke test

### Test Guarantees
- **Autonomous**: No human input required - tests fail if `input()` is called
- **Offline**: Simulator tests work without network/API keys
- **Deterministic**: Seeded random generators for reproducible results
- **Isolated**: Temporary SQLite databases per test

## Safety Warnings (Live Trading)
- Live trading is blocked unless `ENABLE_LIVE_TRADING=true` **and** a valid `LIVE_TRADING_CONFIRMATION_TOKEN` is provided in `.env`.
- Use paper trading by default and verify positions, limits, and order behavior before enabling live trading.
- Review `max_daily_loss_pct`, exposure limits, and `max_trades_per_day` before going live.

## Architecture
```
Scheduler/CLI
    │
    ▼
TradingSystem (main loop)
    │
    ├─ MarketData (Alpaca/Yahoo)
    ├─ BehavioralFeatureCalculator
    ├─ Policy (Rules or LLM)
    ├─ RiskGate + PositionSizer
    ├─ Broker (Simulator or Alpaca)
    └─ Storage (SQLite)
```

At each cycle, the system pulls market data, computes behavioral features, makes policy decisions, applies risk gates and sizing, then submits or simulates orders and persists decisions and executions.

## LLM Provider Selection

The system supports two LLM providers for the AI-based trading policy:

### OpenAI (default)
```yaml
# config.yaml
llm_provider: 'openai'
llm_model: 'gpt-4o-mini'
```
Set `OPENAI_API_KEY` in your environment.

### Gemini (Google)
```yaml
# config.yaml  
llm_provider: 'gemini'
llm_model: 'gemini-2.0-flash'
```
Set `GEMINI_API_KEY` in your environment.

Note: If no API key is set, the system falls back to the rules-based policy.

## Web Monitoring Dashboard

A local FastAPI server provides read-only monitoring endpoints.

### Enabling the Web Server
```yaml
# config.yaml
enable_web: true
web_host: '127.0.0.1'
web_port: 8000
```

### Available Endpoints
- GET /health - Health check
- GET /status - Current system status
- GET /decisions - Recent trading decisions
- GET /orders - Recent orders
- GET /positions - Current positions
- GET /logs/recent - Recent log entries

### Testing the API
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
```

## Discord Notifications

Get real-time alerts via Discord webhook.

### Setup
1. Create a Discord webhook in your server
2. Configure in config.yaml or environment:
```yaml
enable_discord: true
discord_webhook_url: 'https://discord.com/api/webhooks/...'
```
Or set `DISCORD_WEBHOOK_URL` env var.

### Notification Events
- Trade signals submitted
- Orders filled
- Risk gate blocks
- Kill switch triggered
- Large price moves (configurable threshold)

### Configuration Options
```yaml
notify_on_trade: true
notify_on_risk_blocks: true
notify_on_large_move: true
large_move_threshold_pct: 3.0
notify_on_kill_switch: true
```
