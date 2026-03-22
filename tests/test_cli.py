"""Tests for scripts/run_backtest.py utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_backtest import _import_strategy


class TestImportStrategy:
    def test_valid_import(self) -> None:
        cls = _import_strategy("pegasus.strategy.examples.sma_cross.SMACrossStrategy")
        from pegasus.strategy.examples.sma_cross import SMACrossStrategy
        assert cls is SMACrossStrategy

    def test_invalid_module(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            _import_strategy("nonexistent.module.Foo")

    def test_invalid_class(self) -> None:
        with pytest.raises(AttributeError):
            _import_strategy("pegasus.strategy.examples.sma_cross.NonexistentClass")
