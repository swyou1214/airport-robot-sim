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
