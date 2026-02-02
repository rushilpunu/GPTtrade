from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Iterable

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"


class _StubMarketData:
    def get_ohlcv(self, *_args: object, **_kwargs: object) -> None:
        return None

    def get_quote(self, *_args: object, **_kwargs: object) -> None:
        return None


def _strip_comments_and_strings(text: str) -> str:
    tokens: list[str] = []
    for tok_type, tok_str, _start, _end, _line in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok_type in (tokenize.COMMENT, tokenize.STRING):
            tokens.append(" " * len(tok_str))
        else:
            tokens.append(tok_str)
    return "".join(tokens)


def _iter_py_files() -> Iterable[Path]:
    return sorted(SRC_DIR.rglob("*.py"))


def test_no_input_called_during_cycle(mock_no_input: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    def _noop_run_cycle(self: object) -> None:
        return None

    monkeypatch.setattr(main, "_build_market_data", lambda _config: _StubMarketData())
    monkeypatch.setattr(main.TradingSystem, "run_cycle", _noop_run_cycle)

    config_path = str(ROOT_DIR / "config.yaml")
    main.main(["--once", "--config", config_path])


def test_static_scan_no_input_calls() -> None:
    pattern = re.compile(r"\b(input|getpass|prompt|raw_input)\s*\(")
    hits: list[str] = []

    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        cleaned = _strip_comments_and_strings(text)
        for match in pattern.finditer(cleaned):
            line = cleaned.count("\n", 0, match.start()) + 1
            hits.append(f"{path.relative_to(ROOT_DIR)}:{line}: {match.group(0)}")

    if hits:
        joined = "\n".join(hits)
        raise AssertionError(f"Interactive prompts detected in src/:\n{joined}")


def test_no_interactive_imports() -> None:
    interactive_modules = {
        "cmd",
        "code",
        "codeop",
        "pdb",
        "readline",
        "rlcompleter",
    }
    hits: list[str] = []

    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise AssertionError(f"Failed to parse {path}: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    module = name.name.split(".")[0]
                    if module in interactive_modules:
                        hits.append(f"{path.relative_to(ROOT_DIR)}:{node.lineno}: import {name.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".")[0]
                if module in interactive_modules:
                    hits.append(f"{path.relative_to(ROOT_DIR)}:{node.lineno}: from {node.module} import")

    if hits:
        joined = "\n".join(hits)
        raise AssertionError(f"Interactive imports detected in src/:\n{joined}")
