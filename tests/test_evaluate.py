"""Unit tests for evaluate.py's pure aggregation helpers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev


def test_q_table_delta_from_blank():
    after = [[1.0, 2.0], [3.0, 4.0]]
    assert ev.q_table_delta(None, after) == pytest.approx(10.0)


def test_q_table_delta_between_tables():
    before = [[1.0, 1.0], [0.0, 2.0]]
    after = [[2.0, 3.0], [0.0, 2.0]]
    assert ev.q_table_delta(before, after) == pytest.approx(3.0)


def test_q_table_delta_missing_after_is_zero():
    assert ev.q_table_delta([[1.0, 1.0]], None) == 0.0


def test_rolling_success_window():
    assert ev.rolling_success([1, 0, 1, 1], window=2) == \
        pytest.approx([100.0, 50.0, 50.0, 100.0])


def test_rolling_success_short_series_is_cumulative():
    assert ev.rolling_success([1, 0], window=10) == \
        pytest.approx([100.0, 50.0])
