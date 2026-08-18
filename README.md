# Project ARGO — Autonomous Robot Ground Operations

![CI](https://github.com/swyou1214/airport-robot-sim/actions/workflows/ci.yml/badge.svg)

A physics-based simulation of an autonomous luggage-transport robot in
PyBullet. The robot navigates a multi-segment airport taxiway with
moving traffic using pure-pursuit tracking, predictive crossing safety,
and a reactive collision-recovery state machine. A Q-learning agent
learns gap acceptance at crossings from its own experience, persisted
across runs.

![Demo run — 3D view and navigation view side by side, depot to gate.
The light above the robot shows its drive command: green = GO, red =
STOP. Watch both panels at the service-road crossing: the light turns
red, the map banner reads STOP, and the readout drops to 0.0
m/s.](docs/demo.gif)

**Measured performance** (automated evaluation harness, 100 randomized
episodes each):

- **Learning from scratch** (blank Q-table): 100% navigation success,
  mean 25.4s depot-to-gate — with every collision/near-miss
  concentrated in the exploration phase (5 of 6 incident episodes in
  the first 21; zero in the final 50) →
  `evaluation/learning_curve_from-scratch.png`
- **Trained policy**: 100% success, mean 25.6s, 0.01
  collisions/episode → `evaluation/learning_curve_trained-policy.png`

The results show that the agent learns gap acceptance from scratch,
converges to a stable policy and keeps the policy in subsequent runs
whilst the safety layers ensure successful mission completion.

---

## Quick start

```bash
# Python 3.10 virtualenv with pybullet + matplotlib (see install note below)
python robot_sim.py            # watch one run in the GUI (real time)
python evaluate.py             # run 20 headless episodes, print stats,
                               # regenerate evaluation/learning_curve.png
```

Every run of `robot_sim.py` updates `q_table.json`, so the crossing
agent continues learning across both GUI and headless runs. That file
is not tracked in git: a fresh clone starts with a blank Q-table and
trains its own, as shown by the from-scratch curve above. The final
trained table is preserved as `final_q_table` inside
`evaluation/eval_results_trained-policy.json`.

### GUI controls

- **Arrow keys** — move the camera target (forward/backward/strafe)
- **Mouse drag / scroll** — hold Ctrl while dragging to rotate, and
  scroll to zoom (PyBullet defaults)
- Keyboard and mouse control coexist

### Navigation view

A second window ("ARGO - Navigation View") opens alongside the 3D
view: a turn-by-turn style 2D map centred on the robot, showing the
road layout, the planned route, the driven trail, live traffic, a
GO/STOP banner, and distance/ETA/speed to the gate. The robot stays
in the middle and the map slides under it, like a car navigation
display. Run with `SIM_MAP=0` to suppress it.

- **`+` / `−` buttons** (top right, or the `+`/`−` keys) zoom between
  a close follow view and the whole tarmac.

It renders in its own process ([nav_map.py](nav_map.py)) and the
simulation streams it small state messages over a non-blocking pipe —
a matplotlib redraw costs ~60ms, which inline would stall the 480 Hz
loop and break real-time pacing. If the window is closed or falls
behind, updates are dropped and the simulation carries on. The map
never *blocks* the simulation, but the two processes do share the
machine: on integrated graphics, a GUI run measured ~20% slower with
the map open.

### macOS 15 install note

PyBullet's vendored zlib conflicts with the macOS 15 SDK. Fix: download
the PyBullet source, delete line 128 of
`examples/ThirdPartyLibs/zlib/zutil.h`, then `pip install .`

---

## The world

19 road segments form the tarmac (grey roads with dashed centerlines):

```
y=18  ┌─────── apron back ────────┐   ← car, 4.5 m/s
      │  Gate A  Gate B  Gate C   │
y=15  │──── apron midline ────────│   ← car, 5.0 m/s
      │                           │
y=12  └─────── apron front ───────┘   ← car, 3.5 m/s
      |         |         |
y=8   ══════ service road ══════════  ← car, 4.0 m/s  (Q-learning crossing)
      |         |         |
      A         B         C           connectors at x=6, 14, 22
      |         |         |
y=0   ══════ main taxiway ══════════  (depot at x=0 — no traffic;
                                       the robot drives along this road)

Gate bay: walled enclosure y=20–26 centred on x=14 (the destination)
Parked plane: static aircraft to the right of the bay
```

The robot follows a fixed **ideal route** — the literal road centerline
from the depot, along the taxiway, up connector B, and through the
apron into the gate bay — via pure pursuit (1m lookahead).

The two views divide the work: the **3D window** shows what the robot
did (breadcrumb trails, red for robot 1 and blue for robot 2, switching
to cyan on a return leg) against the road network's own dashed
centrelines, while the **navigation map** shows the plan — the route
for the leg currently being driven.

Each car's starting position and direction are **randomized every
run**, so no two episodes present the same traffic timing. Set
`SIM_SEED=<int>` for a reproducible run.

---

## How it works

### Route following
Normal driving never touches a planner: the robot projects itself onto
the ideal route and steers toward a point 1m further along it (pure
pursuit). A* on a 0.5m occupancy grid — constrained to a 1m corridor
around the road network — is used only *reactively*, to plan short
detours after a collision.

### Crossing safety (rule-based, apron roads y=12/15/18)
Before each road the robot holds in a window that ends 1m *before* the
car's lane (once inside the lane it is committed — it never stops in a
car's path). It proceeds only when every car's **predicted time to
reach the crossing** — computed from position, speed, and direction,
including the bounce at the road's end — exceeds 2.5s. A car whose
body currently overlaps the crossing blocks it regardless of
direction.

### Q-learning gap acceptance (service road, y=8)
At the service-road crossing the WAIT/GO decision is *learned*, not
hard-coded:

- **State (10):** gap to the nearest car, in 5 buckets
  (0–3 / 3–5 / 5–7 / 7–9 / 9+ m) × whether that car is **approaching
  or receding** — the same 6m gap is safe behind a receding car and a
  near-miss in front of an approaching one
- **Actions:** WAIT / GO — exactly one decision per approach
- **Reward:** a GO is scored *after* the crossing completes: +10 if no
  car came within 2m of the robot, −10 otherwise. A WAIT earns
  +10 minus 1/s of waiting once traffic clears
- **Learning:** tabular Q-learning (α=0.3, γ=0.9), ε-greedy
  exploration decaying 0.30 → 0.05 across runs
- **Persistence:** `q_table.json`, updated every crossing; the file's
  state layout is validated on load and auto-reset if the schema
  changed

A typical learned table — note the direction split at equal distance:

```
 0-3m appr: WAIT=+6.2  GO=-2.2  -> prefers WAIT
 5-7m rcdg: WAIT= 0.0  GO=+3.0  -> prefers GO
 5-7m appr: WAIT=+5.4  GO=-2.0  -> prefers WAIT
 7-9m appr: WAIT= 0.0  GO=+11.7 -> prefers GO
```

### Round trip (`SIM_RETURN=1`)
With this set, reaching the gate is not the end: the robot turns around
and drives back to the depot by a **different route** — west along the
apron, down the Gate A connector, along the service road, and down
connector A to the taxiway. It shares no horizontal road and no
connector with the outbound leg. A round trip takes ~58–75s against
~25s for the outbound leg alone.

Two of the return legs run *along* a road carrying traffic rather than
across it, which the robot has no overtaking behaviour for. They are
handled differently because the geometry forces it:

- The 6m apron leg runs **offset to y=16.8** rather than on the y=15
  centreline — lane discipline, so the car passes alongside. A gap-based
  reservation is provably unusable there: the robot needs ~4.3s to cover
  6m while the midline car sweeps 20m at 5 m/s, so it always catches up.
  That version deadlocked in every episode tested.
- The 2m service-road leg keeps to the centreline and uses
  `lane_conflict_free()`, which forward-simulates the robot and each car
  (bounces included) and only commits if they never conflict. That *is*
  satisfiable over 2m.

It is opt-in so the documented depot-to-gate measurements above keep
measuring the task they were recorded against.

### Two robots (`SIM_ROBOTS=2`)
The moment the first robot reaches the gate and turns for home, a
second is deployed from the depot on the outbound route — so the bay is
free. Both do the full round trip, and they stay out of phase by
construction: the second is still outbound while the first is coming
home, which is why no dispatcher is needed yet.

Every piece of mutable state lives on a `Robot`, so one body of control
logic drives both: each has its own route, leg, state machine, signal
light, trail, detour bookkeeping and — importantly — its own
Q-crossing latches, so two robots at the service road cannot corrupt
each other's pending decision. The **Q-table is shared**: both learn
into one policy. Robots also treat each other as obstacles, so a bump
triggers the same reverse-and-rejoin recovery a car would.

```bash
SIM_RETURN=1 SIM_ROBOTS=2 python robot_sim.py
```

![Two robots — the second deploys from the depot as the first turns for
home; both round trips run out of phase](docs/two_robots.gif)

Measured over three seeds: both robots complete every time, ~87-93s for
the pair. Two seeds carry one recoverable collision on a return leg --
the same service-road lane-reservation fragility a single robot has, not
a robot-robot conflict (it occurs after the other robot has finished).

Both viewers show the fleet. The navigation map draws an arrow per robot
(revealed as they deploy) coloured by that robot's GO/STOP signal, plus
a trail per robot — **red for robot 1, blue for robot 2**, matching the
3D view. The map keeps a steady framing on the lead robot at whatever
zoom you choose; other robots are drawn when they fall inside it and
never pull the view around.

**Trails show only the leg in progress.** Reaching a destination wipes
that robot's trail, so what you see is the drive to its *next* stop
rather than an accumulating tangle. That takes clearing four separate
stores — the GUI's debug lines, capture mode's disc markers (debug lines
don't render offscreen, so recordings use discs), the live map process,
and the capture panel's own point list.

### Collision recovery
Contact with any car or wall triggers a state machine: stop →
**REVERSING** (back off 1.2m) → **REJOINING** (A* detour back onto the
ideal route) → normal driving. During a detour the rule-based crossing
guard stays active (including for the service road, since the Q-agent
isn't consulted mid-detour). Static obstacles (cones, if placed) are
additionally remembered in `learned_obstacles.json` and avoided from
the start of future runs; car/wall contacts are deliberately *not*
persisted — cars move, so their impact location is stale immediately.

---

## Evaluation harness

```bash
python evaluate.py                  # 20 episodes, keep learning
python evaluate.py -n 50            # more episodes
python evaluate.py --fresh          # wipe q_table.json first
python evaluate.py --seed-start 1   # reproducible seeds 1..N
python evaluate.py --time-limit 180 # per-episode sim-time cap
python evaluate.py --tag baseline   # label outputs: learning_curve_baseline.png
```

Each episode runs `robot_sim.py` headless in a fresh subprocess and
reports success, navigation time, collisions, near-misses, and
decisions, then the harness aggregates everything into:

- `evaluation/learning_curve.png` — navigation time, trailing success rate,
  Q-table convergence (Σ|ΔQ| per episode), and ε decay
- `evaluation/eval_results.json` — raw per-episode records + final Q-table

Environment hooks (used by the harness, available manually too):

| Variable | Effect |
|---|---|
| `SIM_HEADLESS=1` | no GUI (`p.DIRECT`), no real-time pacing |
| `SIM_TIME_LIMIT=<s>` | abort the episode as a failure after *s* sim-seconds |
| `SIM_SEED=<int>` | reproducible traffic + exploration |
| `SIM_CAPTURE=<path.gif>` | render the run offscreen and write an animated GIF |
| `SIM_MAP=0` | suppress the 2D navigation map window |
| `SIM_RETURN=1` | after the gate, drive back to the depot by a different route |
| `SIM_ROBOTS=2` | deploy a second robot when the first turns for home (needs `SIM_RETURN`) |
| `SIM_CROSSING=rule` | hand the service-road crossing to the rule baseline |
| `SIM_FREEZE_Q=1` | greedy evaluation: no exploration, no Q updates |

The README demo was captured with:

```bash
SIM_HEADLESS=1 SIM_SEED=205 SIM_CAPTURE=docs/demo.gif python robot_sim.py
```

`SIM_CAPTURE` renders both panels offscreen — the 3D scene through
PyBullet's software renderer, and the navigation map through
`nav_map.py`'s own drawing code, so the recorded map can never drift
from the live one.

---

## Project structure

```
airport-robot/
├── robot_sim.py            # the complete simulation (~2,490 lines)
├── nav_map.py              # 2D navigation map window (separate process)
├── evaluate.py             # evaluation harness (~480 lines)
├── q_table.json            # learned crossing policy (auto-generated, untracked)
├── learned_obstacles.json  # static-obstacle memory (auto-generated, untracked)
├── evaluation/             # eval records + learning curves (auto-generated)
├── ab_comparison/          # A/B records + comparison figure (auto-generated)
├── docs/                   # captured runs: demo.gif, two_robots.gif
├── README.md
└── PROJECT_ARGO_SUMMARY.md # technical summary for development
```

## Tuning

All driving constants live near the top of the relevant sections in
`robot_sim.py`:

```python
MAX_SPEED = 10.0           # wheel angular velocity (≈1.7 m/s linear)
WHEEL_FORCE = 35           # motor torque limit
TURN_GAIN = 8              # steering aggressiveness
LOOKAHEAD_DISTANCE = 1.0   # pure-pursuit lookahead (m)
REVERSE_DISTANCE = 1.2     # back-off after collision (m)
MIN_CROSSING_TIME = 2.5    # predictive crossing threshold (s)
NEAR_MISS_DISTANCE = 2.0   # Q-reward near-miss radius (m)
GAP_BUCKETS = [3, 5, 7, 9] # Q-state distance boundaries (m)
```

Physics steps at 480 Hz (`p.setTimeStep(1/480)`), matched to the main
loop's bookkeeping so one loop iteration is 1/480s everywhere and the
GUI paces in real time.

## Tests & CI

```bash
python -m pytest -v        # 17 tests, ~2 min locally
```

- **Unit tests** — `nav_map`'s geometry (marker rotation, road
  polygons, view centring, zoom limits) and `evaluate`'s aggregation
  helpers (Q-table deltas, rolling success rate)
- **End-to-end tests** — full headless episodes of `robot_sim.py`
  (seeded and unseeded) asserting the robot reaches the gate; the round
  trip returning to the depot and the fleet deploying a second robot,
  each with a check that the feature stays opt-in; and a capped capture
  run asserting the composite GIF pipeline produces a valid two-panel
  animation

GitHub Actions runs the same suite on every push and pull request
(`.github/workflows/ci.yml`). `robot_sim.py`'s internals (A*,
crossing prediction, Q-updates) are covered through the end-to-end
episodes; direct unit tests for them await the planned module split.

## Known limitations

- The Q-state encodes gap + direction but not car speed, and only the
  service-road crossing is learned — the apron roads use the fixed
  predictive rule
- No same-lane traffic handling (the taxiway is kept car-free by
  design; there is no overtaking behavior)
- Two robots run without a dispatcher: they stay out of phase by
  construction, so they never contest the shared taxiway. A third
  robot, or different deployment timing, would need real coordination
- The robot reads car positions from the physics engine directly — no
  simulated sensors or noise
- With ε floored at 0.05, rarely-visited Q-states fill in slowly
