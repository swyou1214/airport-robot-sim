"""Unit tests for evaluate.py's pure helpers (aggregation, Q-table loading)."""
import json
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


# ---- load_q_source: what `--ab --q-from` evaluates ----

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def test_load_q_source_reads_q_table_file(tmp_path):
    f = write_json(tmp_path / "q.json", {"q_table": [[1.0, 2.0], [0, 3]],
                                         "epsilon": 0.05})
    assert ev.load_q_source(f) == [[1.0, 2.0], [0, 3]]


def test_load_q_source_reads_committed_eval_record():
    """The committed trained-policy record is what `--ab --q-from` points
    at on a fresh clone, so it must stay loadable."""
    table = ev.load_q_source(
        ROOT / "evaluation" / "eval_results_trained-policy.json")
    assert len(table) == 10
    assert all(len(row) == 2 for row in table)


@pytest.mark.parametrize("data", [
    {"episodes": []},                       # no table at all
    {"q_table": []},                        # empty
    {"q_table": [[1.0, 2.0, 3.0]]},         # wrong action count
    {"q_table": [["a", "b"]]},              # not numbers
    {"q_table": [[0.0, 0.0], [0.0, 0.0]]},  # untrained
    [[1.0, 2.0]],                           # bare list, no wrapper
])
def test_load_q_source_rejects_unusable_tables(tmp_path, data):
    f = write_json(tmp_path / "bad.json", data)
    with pytest.raises(ValueError):
        ev.load_q_source(f)


def test_load_q_source_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        ev.load_q_source(tmp_path / "nope.json")
