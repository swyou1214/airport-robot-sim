# Project ARGO — Development Summary
*Technical summary for development continuation. Supersedes the
2026-07 handoff version (which described the pre-evaluation-harness
architecture).*

---

## What this project is

A physics-based simulation of an autonomous airport luggage-transport
robot built in PyBullet (Python 3.10). The robot navigates a
multi-segment airport tarmac from a depot to a gated jetbridge bay,
crossing randomized moving traffic, recovering from collisions, and
*learning* gap acceptance at one crossing via Q-learning.

Goals:
1. A portfolio piece demonstrating autonomous navigation, RL, and
   simulation-first development for ML DevEx / robotics engineering roles
2. A working simulation extensible toward production-grade autonomous
   ground vehicle behaviour

---

## Repository structure

```
airport-robot/
├── robot_sim.py            # Main simulation (~1,760 lines)
├── evaluate.py             # Evaluation harness (~300 lines)
├── q_table.json            # Q-learning state, persists across runs (auto-generated)
├── learned_obstacles.json  # Static-obstacle memory (auto-generated)
├── eval_results*.json      # Evaluation records (auto-generated; --tag names)
├── learning_curve*.png     # Evaluation figures (auto-generated)
├── README.md               # User-facing documentation
└── PROJECT_ARGO_SUMMARY.md # This file
```

## Tech stack

- **PyBullet 3.2.7** — physics engine and renderer, stepped at 480 Hz
  (`p.setTimeStep(1/480)`, matched to the loop's 1/480 bookkeeping so
  the GUI runs in real time)
- **Python 3.10** in `venv310`
- **Matplotlib** — used by the evaluation harness for learning curves
- No external ML libraries — all algorithms implemented from scratch

**macOS 15 install note:** PyBullet's vendored zlib conflicts with the
macOS 15 SDK. Fix: download source, delete line 128 of
`examples/ThirdPartyLibs/zlib/zutil.h`, then `pip install .`

---

## Road network (ROAD_SEGMENTS)

19 segments forming the tarmac layout:

```
y=18  ┌─────── apron back ────────┐   ← car, 4.5 m/s   (rule-based)
      │  Gate A  Gate B  Gate C   │
y=15  │──── apron midline ────────│   ← car, 5.0 m/s   (rule-based)
      │                           │
y=12  └─────── apron front ───────┘   ← car, 3.5 m/s   (rule-based)
      |         |         |
y=8   ══════ service road ══════════  ← car, 4.0 m/s   (Q-LEARNING)
      |         |         |
      A         B         C           cross-connectors at x=6, 14, 22
      |         |         |
y=0   ══════ main taxiway ══════════  (depot at x=0 — NO traffic)

Gate bay: walled enclosure y=20–26, centred x=14 (destination)
Parked plane: static aircraft right of the bay
```

**Key coordinates:** `start_pos=(0,0)`, `target_pos=(14,23)`; grid
66×62 at 0.5m, origin (−2,−3); `ROAD_HALF_WIDTH=1.0`.

**The main taxiway is deliberately car-free**: the robot drives ALONG
it (depot → connector-B turn at x=14), and the robot has no same-lane
avoidance behaviour, only after-contact recovery. All traffic lives on
roads the robot crosses *perpendicular* to.

**Traffic randomization:** each car's start position and direction are
randomized per run (`random.uniform` over its road span). `SIM_SEED`
env var seeds both traffic and ε-exploration for reproducible runs.

---

## ideal_route (the yellow line)

```python
ideal_route = [(0,0), (14,0), (14,8), (14,12), (14,15), (14,20), (14,23)]
```

Drawn once as a gold line; never changes. Tracked with
`lookahead_point_on_route()` (pure pursuit, 1m lookahead). A* +
`smooth_path` are used only for reactive post-collision detours.

---

## State machine

```
STATE_DRIVING   Pure-pursuit tracking of ideal_route
     │
     ├──[contact: cone/car/wall]──► STATE_REVERSING
     │                                  │
     │                             [1.2m backed up]
     │                                  │
     │                             STATE_REJOINING ──[detour done]──► DRIVING
     │                             (rule-based crossing guard stays
     │                              active, INCL. the service road)
     │
     ├──[Q-agent: WAIT]──► STATE_WAITING ──[predicted-time clear]──► DRIVING
     │
     └──[rule-based hold]── wheels stopped, re-check next frame
```

---

## Crossing safety

### Rule-based (apron roads y=12, 15, 18) — predict-and-wait
- Hold window per road: `[road.y − 2.0, road.y − 1.0]` — ends 1m
  BEFORE the car's lane. Inside 1m the robot is committed and is never
  told to stop in the lane (the old window extended 0.5m past the road
  centre, which could freeze the robot in a car's path).
- Proceed only when `car_time_to_crossing()` ≥ `MIN_CROSSING_TIME`
  (2.5s) for every car on the road. The ETA is direction-aware: an
  approaching car counts its straight-line time; a receding car counts
  the full path through its end-of-road bounce and back. A car whose
  body currently overlaps the crossing point returns ETA 0 regardless
  of direction (its tail can hang over the crossing while "receding").
- A pure distance threshold is geometrically impossible here: on the
  x=4–24 apron roads the maximum possible edge gap is 9.2m, so any
  requirement above that deadlocks. That's why the predictive check
  replaced the old `safe_gap` distances.

### Q-learning (service road y=8, crossing at x=14)
The robot climbs connector B and crosses this road perpendicular to
traffic — a genuine gap-acceptance decision, made by the agent:

```
State:   10 = 5 gap buckets [0–3, 3–5, 5–7, 7–9, 9+ m]
              × car direction {receding, approaching}
         encoding: q_state = bucket*2 + (1 if approaching)
Actions: 0=WAIT, 1=GO — ONE decision per approach (latched via
         q_go_active / q_crossing_done; no per-frame re-decisions)
Reward:  GO  — resolved AFTER the robot clears the zone, from the
               closest any car actually came: +10 if ≥ 2.0m
               (NEAR_MISS_DISTANCE), −10 otherwise
         WAIT — +10 − 1/s waited, granted when the predicted-time
               check clears the road (direction-aware release)
α=0.3, γ=0.9, ε=0.30 ×0.85/run, floor 0.05
Q-table: q_table.json (10×2); state-layout validated on load and
         auto-reset with a notice if the schema changed
```

The direction bit is what makes the middle buckets learnable: a 6m gap
behind a receding car is safe (GO), the same gap in front of an
approaching car is a near-miss (WAIT). Decision zone: |x−14| < 1.5,
y ∈ (6, 9); the GO reward resolves on zone exit.

---

## Collision recovery

Two obstacle classes, both feeding the same reverse-and-rejoin flow:

- **Static (cones, if placed):** location is learned, appended to
  `learned_obstacles.json`, marked dark red on the floor next run, and
  its grid cells are blocked for future planning. (No cones are placed
  in the current scene; the machinery is exercised via cars/walls.)
- **Transient (cars, bay walls):** trigger recovery but are NOT
  persisted or added to the blocked map — a car's position at impact
  is stale immediately, and the walls sit outside the road corridor.

Recovery: stop → reverse 1.2m → `find_rejoin_target()` (two-phase
forward walk along the route, +1m clearance) → A* detour
(`runtime_blocked`) → REJOINING → DRIVING. Per-body collision cooldown
2.0s.

---

## Evaluation harness (evaluate.py)

Runs N headless episodes as fresh subprocesses, parses each episode's
machine-readable `RESULT` line, snapshots `q_table.json` after every
episode, and writes `eval_results.json` + `learning_curve.png`
(navigation time, trailing success rate, Σ|ΔQ| convergence, ε decay).

```
python evaluate.py [-n N] [--fresh] [--seed-start K] [--time-limit S] [--tag NAME]
```

Sim-side hooks (env vars, all no-ops in normal GUI use):
`SIM_HEADLESS=1` (p.DIRECT, no sleep), `SIM_TIME_LIMIT` (abort as
failure), `SIM_SEED` (reproducibility), `SIM_CAPTURE=<path.gif>`
(offscreen GIF capture: route/trail drawn as real geometry since debug
lines don't render offscreen, robot marked with a beacon sphere,
frames via TinyRenderer at 10 fps, played back 2x). Every run prints
`RESULT success=… sim_time=… collisions=… near_misses=… go=… wait=…
epsilon=… run=…`.

**Measured baselines (2026-07-11), the portfolio pair:**
- **Learning from scratch** — 100 unseeded episodes, blank table,
  ε 0.30→0.05 (`learning_curve_from-scratch.png`): 100% success, mean
  25.4s, 1 collision + 6 near-misses total, with 5 of 6 incident
  episodes inside the first 21 and **zero incidents in the final 50**.
  Early episodes update the table every crossing (Σ|ΔQ| ≈ 3 plateau);
  zero-update episodes become common late.
- **Trained policy** — 100 unseeded episodes on the pre-trained table
  (`learning_curve_trained-policy.png`): **100% success, mean 25.6s /
  median 25.5s (21.4–31.4s), 1 collision (0.01/ep, exploratory),
  4 near-misses, 53 GO / 41 WAIT**. All 8 reachable decision states
  learned; policy stable (Σ|ΔQ| oscillates ~1.2 from varying WAIT
  rewards + ε-exploration, argmax unchanged). The 9m+ states are
  structurally unreachable as decision states (the agent only engages
  under 9m).

---

## Architecture changes vs. the 2026-07 handoff doc

If you last saw the old summary, these are the deltas:

1. **Physics timestep fixed** — was 1/240 vs. the loop's 1/480
   bookkeeping (cars effectively half-speed); now 480 Hz throughout.
2. **Traffic randomized** — was fully deterministic (one scenario per
   layout); now random start/direction per run + `SIM_SEED`.
3. **Q-crossing moved** from the main taxiway (y=0, which had no cars
   — the agent never engaged) to the service road (y=8). The taxiway
   stays car-free by design.
4. **Q-state gained the direction bit** (5 → 10 states) and the GO
   reward became outcome-based (resolved after the crossing from
   actual closest approach) instead of a threshold check at decision
   time; decisions are latched to one per approach (previously
   re-decided and file-saved every frame at 480 Hz).
5. **Rule crossings upgraded to predict-and-wait** (bounce-aware ETA +
   occupancy check + commit margin), replacing the direction-blind
   `safe_gap` distances; the apron-back road (y=18) is now guarded
   (it previously wasn't); the guard also runs during REJOINING.
6. **Collision recovery actually fires** — it previously only watched
   cones (of which there are none); it now watches cars and bay walls
   too, without persisting them.
7. **Evaluation harness added** (evaluate.py + RESULT line + env
   hooks); success rate is now measured, not estimated.

---

## Known issues / limitations

1. **Q-state has no car-speed feature** and covers only the service
   road; the apron crossings are fixed-rule. Multi-crossing learning
   would need per-road or shared state.
2. **No same-lane traffic handling** — no overtaking; the taxiway is
   kept car-free by design.
3. **Single agent** — a second robot needs conflict resolution /
   dispatcher at shared connectors.
4. **No sensor simulation** — perfect state via PyBullet API.
5. **`smooth_path` can still cut corners** where short apron segments
   are geometrically close.
6. **ε floor (0.05) fills rare states slowly** — a harness `--explore`
   flag (temporarily raised ε) would speed up covering the remaining
   zero-valued states.
7. **"7–9m receding" prefers WAIT only because GO was never sampled
   there** at ε=0.05 — harmless (the predictive release crosses almost
   immediately anyway), but it's the visible case for the `--explore`
   flag.

---

## Driving constants (easy to tune)

```python
MAX_SPEED = 10.0            # wheel angular velocity (≈1.7 m/s linear)
WHEEL_FORCE = 35            # motor torque limit
TURN_GAIN = 8               # steering aggressiveness
TURN_SPEED_FLOOR = 0.15     # min speed fraction during sharp turns
LOOKAHEAD_DISTANCE = 1.0    # pure-pursuit lookahead (m)
REVERSE_DISTANCE = 1.2      # post-collision back-off (m)
REVERSE_SPEED = -8
ARRIVAL_RADIUS = 0.3
MIN_CROSSING_TIME = 2.5     # predictive crossing threshold (s)
CROSSING_STOP_MARGIN = 2.0  # hold window start before a road (m)
CROSSING_COMMIT_MARGIN = 1.0# hold window end / commit line (m)
NEAR_MISS_DISTANCE = 2.0    # Q-reward near-miss radius (m)
GAP_BUCKETS = [3, 5, 7, 9]  # Q-state distance boundaries (m)
```

---

## Suggested next steps

### High priority
- [ ] **Harness `--explore` flag** — temporarily raise ε to fill the
      remaining zero Q-states, then re-freeze
- [ ] **Multi-crossing Q-learning** — extend the agent to the apron
      roads (shared table keyed by road, or add car speed to state)
- [x] **Larger evaluation batches** — 100-episode trained-policy run
      complete (see baselines above); still open: fresh-vs-trained
      comparison curves

### Medium priority
- [ ] **Second robot** on another route (depot → Gate A) + dispatcher
- [ ] **Simulated sensor noise** on car positions
- [ ] **Live dashboard** — matplotlib window with Q-table heatmap
      updating during a GUI run

### Lower priority
- [ ] **Gate selection** — runtime gate assignment (A/B/C)
- [ ] **Unit tests** for a_star, smooth_path, has_clear_line,
      find_rejoin_target, car_time_to_crossing, q_state
- [x] **Screen recording / GIF** — `docs/demo.gif` (seed 205: clean
      run with a learned WAIT at the service-road crossing), captured
      via the `SIM_CAPTURE` hook
