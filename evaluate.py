#!/usr/bin/env python3
"""Evaluation harness for the ARGO airport-robot simulation.

Runs N headless episodes of robot_sim.py (each as a fresh subprocess,
so PyBullet state can't leak between runs), parses each episode's
RESULT line, snapshots the persisted Q-table after every episode, and
reports:

  - success rate, and mean/median navigation time over successes
  - collision / near-miss / GO-WAIT decision totals
  - a learning-curve figure        -> learning_curve.png
  - raw per-episode records        -> eval_results.json

The Q-table (q_table.json) persists across episodes -- that's the
point: the harness measures how the agent improves as it accumulates
crossing experience.

Usage:
  python evaluate.py                  # 20 episodes, keep learning
  python evaluate.py -n 50            # more episodes
  python evaluate.py --fresh          # wipe q_table.json first
  python evaluate.py --seed-start 1   # reproducible seeds 1..N
  python evaluate.py --time-limit 180 # per-episode sim-time cap (s)
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
SIM_FILE = os.path.join(BASE, "robot_sim.py")
Q_FILE = os.path.join(BASE, "q_table.json")
RESULTS_JSON = os.path.join(BASE, "eval_results.json")
CURVE_PNG = os.path.join(BASE, "learning_curve.png")

# Palette (validated defaults): ink/chrome + one categorical hue for
# data series, status-critical for failure marks. Single-hue on
# purpose -- each panel shows one measure, so identity never rides on
# color alone.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"     # blue, categorical slot 1
FAILURE = "#d03b3b"    # status: critical


def run_episode(time_limit, seed=None):
    """Run one headless episode of robot_sim.py; return its parsed
    RESULT record (or a failure record if the sim crashed)."""
    env = os.environ.copy()
    env["SIM_HEADLESS"] = "1"
    env["SIM_TIME_LIMIT"] = str(time_limit)
    if seed is not None:
        env["SIM_SEED"] = str(seed)
    else:
        env.pop("SIM_SEED", None)

    proc = subprocess.run([sys.executable, SIM_FILE],
                          capture_output=True, text=True,
                          env=env, timeout=1800)
    match = re.search(r"^RESULT (.+)$", proc.stdout, re.MULTILINE)
    if not match:
        tail = (proc.stdout + proc.stderr).strip()[-400:]
        return {"success": 0, "sim_time": 0.0, "collisions": 0,
                "near_misses": 0, "go": 0, "wait": 0,
                "epsilon": 0.0, "run": 0, "crashed": True, "error": tail}

    record = {}
    for pair in match.group(1).split():
        key, value = pair.split("=")
        record[key] = float(value) if "." in value else int(value)
    record["crashed"] = False
    return record


def load_q_snapshot():
    """Current persisted Q-table, or None if it doesn't exist yet."""
    try:
        with open(Q_FILE) as f:
            return json.load(f)["q_table"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def q_table_delta(before, after):
    """Total absolute Q-value change between two table snapshots --
    the per-episode convergence signal (it shrinks as learning
    settles). A missing 'before' means the table was fresh zeros."""
    if after is None:
        return 0.0
    if before is None:
        before = [[0.0] * len(row) for row in after]
    return sum(abs(a - b)
               for row_a, row_b in zip(after, before)
               for a, b in zip(row_a, row_b))


def rolling_success(successes, window=10):
    """Success rate (%) over a trailing window, per episode."""
    rates = []
    for i in range(len(successes)):
        lo = max(0, i - window + 1)
        chunk = successes[lo:i + 1]
        rates.append(100.0 * sum(chunk) / len(chunk))
    return rates


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_curves(records, deltas, out_path, note=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping learning_curve.png")
        return False

    episodes = list(range(1, len(records) + 1))
    successes = [r["success"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.patch.set_facecolor(SURFACE)
    (ax_time, ax_rate), (ax_delta, ax_eps) = axes

    # Panel 1: navigation time per episode; failures marked separately.
    ok_eps = [e for e, r in zip(episodes, records) if r["success"]]
    ok_times = [r["sim_time"] for r in records if r["success"]]
    bad_eps = [e for e, r in zip(episodes, records) if not r["success"]]
    bad_times = [r["sim_time"] for r in records if not r["success"]]
    ax_time.plot(ok_eps, ok_times, color=SERIES, linewidth=2,
                 marker="o", markersize=4, label="success")
    if bad_eps:
        ax_time.scatter(bad_eps, bad_times, color=FAILURE, marker="x",
                        s=45, linewidths=2, label="failure", zorder=3)
        ax_time.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    ax_time.set_title("Navigation time per episode", fontsize=10)
    ax_time.set_ylabel("sim seconds")

    # Panel 2: trailing success rate.
    rates = rolling_success(successes)
    ax_rate.plot(episodes, rates, color=SERIES, linewidth=2)
    ax_rate.set_ylim(-4, 104)
    ax_rate.set_title("Success rate (trailing 10 episodes)", fontsize=10)
    ax_rate.set_ylabel("%")
    ax_rate.annotate(f"{rates[-1]:.0f}%", (episodes[-1], rates[-1]),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=8, color=INK_SECONDARY)

    # Panel 3: Q-table convergence (total |change| per episode).
    ax_delta.plot(episodes, deltas, color=SERIES, linewidth=2,
                  marker="o", markersize=3)
    ax_delta.set_title("Q-table change per episode (convergence)", fontsize=10)
    ax_delta.set_ylabel("sum |dQ|")
    ax_delta.set_xlabel("episode")

    # Panel 4: exploration rate.
    ax_eps.plot(episodes, [r["epsilon"] for r in records],
                color=SERIES, linewidth=2)
    ax_eps.set_title("Exploration rate (epsilon)", fontsize=10)
    ax_eps.set_xlabel("episode")

    for ax in (ax_time, ax_rate, ax_delta, ax_eps):
        style_axis(ax)

    n_ok = sum(successes)
    title = "ARGO evaluation -- Q-learning gap acceptance"
    if note:
        title += f" ({note})"
    fig.suptitle(title, fontsize=13, color=INK, y=0.99)
    fig.text(0.5, 0.945,
             f"{len(records)} episodes | {n_ok}/{len(records)} successful"
             + (f" | mean navigation {statistics.mean(ok_times):.1f}s"
                if ok_times else ""),
             ha="center", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run N headless evaluation episodes of robot_sim.py")
    parser.add_argument("-n", "--episodes", type=int, default=20)
    parser.add_argument("--fresh", action="store_true",
                        help="delete q_table.json first (start learning "
                             "from scratch)")
    parser.add_argument("--seed-start", type=int, default=None,
                        help="seed episodes with SEED_START..SEED_START+N-1 "
                             "for reproducible traffic (default: unseeded)")
    parser.add_argument("--time-limit", type=float, default=240.0,
                        help="per-episode simulated-time cap in seconds "
                             "(default 240)")
    parser.add_argument("--tag", default=None,
                        help="label this run: outputs become "
                             "eval_results_<tag>.json / "
                             "learning_curve_<tag>.png and the tag "
                             "appears in the figure title")
    args = parser.parse_args()

    results_json = RESULTS_JSON
    curve_png = CURVE_PNG
    if args.tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "-", args.tag)
        results_json = os.path.join(BASE, f"eval_results_{safe_tag}.json")
        curve_png = os.path.join(BASE, f"learning_curve_{safe_tag}.png")

    if args.fresh and os.path.exists(Q_FILE):
        os.remove(Q_FILE)
        print("Removed q_table.json -- starting from a blank Q-table.")

    records, deltas = [], []
    q_before = load_q_snapshot()

    print(f"Running {args.episodes} headless episodes "
          f"(time limit {args.time_limit:.0f}s each)...\n")
    for i in range(args.episodes):
        seed = None if args.seed_start is None else args.seed_start + i
        record = run_episode(args.time_limit, seed)
        q_after = load_q_snapshot()
        delta = q_table_delta(q_before, q_after)
        q_before = q_after

        records.append(record)
        deltas.append(delta)

        if record["crashed"]:
            print(f"ep {i + 1:>3}  CRASH  {record['error'][:120]}")
            continue
        status = "ok  " if record["success"] else "FAIL"
        seed_txt = f"seed={seed}" if seed is not None else "unseeded"
        print(f"ep {i + 1:>3}  {status}  t={record['sim_time']:6.1f}s  "
              f"collisions={record['collisions']}  "
              f"near_misses={record['near_misses']}  "
              f"go={record['go']} wait={record['wait']}  "
              f"eps={record['epsilon']:.2f}  dQ={delta:5.1f}  ({seed_txt})")

    # ---- Aggregate summary ----
    n = len(records)
    ok = [r for r in records if r["success"]]
    ok_times = [r["sim_time"] for r in ok]
    print("\n==== Evaluation summary ====")
    print(f"Episodes:        {n}")
    print(f"Success rate:    {len(ok)}/{n} ({100.0 * len(ok) / n:.1f}%)")
    if ok_times:
        print(f"Navigation time: mean {statistics.mean(ok_times):.1f}s  "
              f"median {statistics.median(ok_times):.1f}s  "
              f"min {min(ok_times):.1f}s  max {max(ok_times):.1f}s")
    print(f"Collisions:      {sum(r['collisions'] for r in records)} total "
          f"({sum(r['collisions'] for r in records) / n:.2f}/episode)")
    print(f"Near misses:     {sum(r['near_misses'] for r in records)}")
    print(f"Q decisions:     {sum(r['go'] for r in records)} GO / "
          f"{sum(r['wait'] for r in records)} WAIT")

    summary = {
        "episodes": n,
        "successes": len(ok),
        "success_rate": len(ok) / n,
        "mean_time_success": statistics.mean(ok_times) if ok_times else None,
        "median_time_success": (statistics.median(ok_times)
                                if ok_times else None),
        "total_collisions": sum(r["collisions"] for r in records),
        "total_near_misses": sum(r["near_misses"] for r in records),
        "seed_start": args.seed_start,
        "time_limit": args.time_limit,
        "tag": args.tag,
    }
    with open(results_json, "w") as f:
        json.dump({"summary": summary, "episodes": records,
                   "q_deltas": deltas, "final_q_table": q_before}, f,
                  indent=2)
    print(f"\nRaw records    -> {os.path.basename(results_json)}")

    if plot_curves(records, deltas, curve_png, note=args.tag):
        print(f"Learning curve -> {os.path.basename(curve_png)}")


if __name__ == "__main__":
    main()
