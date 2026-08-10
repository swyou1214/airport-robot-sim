"""End-to-end tests: run the real simulation headless as a subprocess
and assert on its machine-readable RESULT line.

robot_sim.py is a script that executes at import, so its internals are
exercised here the same way evaluate.py drives them -- a full episode
covers route following, crossing safety, the Q-agent, and persistence
in one pass, on every push.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_sim(extra_env, timeout=900):
    env = os.environ.copy()
    env["SIM_HEADLESS"] = "1"
    env.update(extra_env)
    proc = subprocess.run([sys.executable, str(ROOT / "robot_sim.py")],
                          capture_output=True, text=True, env=env,
                          cwd=str(ROOT), timeout=timeout)
    assert proc.returncode == 0, f"sim crashed:\n{proc.stderr[-800:]}"
    match = re.search(r"^RESULT (.+)$", proc.stdout, re.MULTILINE)
    assert match, f"no RESULT line in output:\n{proc.stdout[-800:]}"
    return dict(pair.split("=") for pair in match.group(1).split())


def test_full_episode_reaches_gate():
    result = run_sim({"SIM_SEED": "205", "SIM_TIME_LIMIT": "150"})
    assert result["success"] == "1"
    assert float(result["sim_time"]) < 150


def test_random_traffic_episode_reaches_gate():
    # unseeded traffic: exercises the crossing logic under fresh timing
    result = run_sim({"SIM_TIME_LIMIT": "150"})
    assert result["success"] == "1"


def test_round_trip_returns_to_depot():
    """SIM_RETURN drives the gate leg and then a different route home."""
    result = run_sim({"SIM_SEED": "205", "SIM_RETURN": "1",
                      "SIM_TIME_LIMIT": "220"})
    assert result["success"] == "1"
    assert result["leg"] == "RETURN"
    # the return leg is real travel, not an instant finish
    assert float(result["sim_time"]) > float(result["gate_time"]) + 15


def test_return_leg_is_opt_in():
    """Without SIM_RETURN the episode still ends at the gate, so the
    documented depot-to-gate baselines keep measuring the same task."""
    result = run_sim({"SIM_SEED": "205", "SIM_TIME_LIMIT": "150"})
    assert result["leg"] == "OUTBOUND"
    assert float(result["sim_time"]) == pytest.approx(
        float(result["gate_time"]), abs=0.01)


def test_second_robot_deploys_on_the_return_leg():
    """SIM_ROBOTS=2 puts a second robot on the tarmac when the first
    turns for home, and both complete their missions."""
    result = run_sim({"SIM_SEED": "205", "SIM_RETURN": "1",
                      "SIM_ROBOTS": "2", "SIM_TIME_LIMIT": "250"})
    assert result["success"] == "1"
    assert result["robots"] == "2"
    # each robot decides its own crossing, so the fleet makes two
    assert int(result["go"]) + int(result["wait"]) >= 2


def test_fleet_is_opt_in():
    """Without SIM_ROBOTS the run is a single robot, so every recorded
    baseline keeps measuring what it measured."""
    result = run_sim({"SIM_SEED": "205", "SIM_TIME_LIMIT": "150"})
    assert result["robots"] == "1"
    assert float(result["sim_time"]) == pytest.approx(24.47, abs=0.05)


def test_capture_writes_composite_gif(tmp_path):
    gif = tmp_path / "ci_demo.gif"
    # A short capped run is enough to exercise the whole capture path:
    # TinyRenderer 3D panel + nav_map's Agg map panel + GIF encoding.
    run_sim({"SIM_SEED": "7", "SIM_TIME_LIMIT": "6",
             "SIM_CAPTURE": str(gif)})
    assert gif.exists()
    assert gif.stat().st_size > 10_000
    from PIL import Image
    im = Image.open(gif)
    assert im.n_frames >= 10
    # composite frame: 3D panel + divider + map panel, wider than tall
    assert im.size[0] > im.size[1] * 2
