# Project ARGO — Development Summary
*Technical summary for development continuation. Last updated
2026-11. Supersedes the 2026-07 handoff version (which described the
pre-evaluation-harness architecture).*

---

## What this project is

A physics-based simulation of an autonomous airport luggage-transport
robot built in PyBullet (Python 3.10). The robot navigates a
multi-segment airport tarmac from a depot to a gated jetbridge bay,
crossing randomized moving traffic, recovering from collisions, and
*learning* gap acceptance at one crossing via Q-learning.

**The problem it addresses:** hand-tuned crossing rules ("go only if
every car is ≥ T seconds away") must be tuned for the worst case, so
they are safe but slow — they hold the robot for gaps that were never
dangerous. Distance alone can't separate a safe gap (behind a receding
car) from a near-miss (in front of an approaching one). ARGO tests
whether a learned policy can cut that waiting without giving up safety.

**Headline result** (paired A/B, 40 seeds, identical traffic per pair —
see *Round trip + A/B* below): learned policy and rule both 40/40
success with 0 collisions / 0 near-misses; mean crossing hold 0.49s vs
1.06s, and 0.85s vs 1.83s (−54%) on the 23 contested seeds. The README
now leads with this framing (*The problem* / *The result* sections).

Goals:
1. A portfolio piece demonstrating autonomous navigation, RL, and
   simulation-first development for ML DevEx / robotics engineering roles
2. A working simulation extensible toward production-grade autonomous
   ground vehicle behaviour

---

## Repository structure

```
airport-robot/
├── robot_sim.py            # Main simulation (~2,490 lines)
├── nav_map.py              # 2D navigation map window, separate process (~390 lines)
├── evaluate.py             # Evaluation harness + paired A/B (~480 lines)
├── requirements.txt        # pybullet 3.2.7, numpy, matplotlib, pillow
├── q_table.json            # Q-learning state, persists across runs (auto-generated, gitignored)
├── learned_obstacles.json  # Static-obstacle memory (auto-generated, gitignored)
├── evaluation/             # Eval records + learning curves (committed portfolio artifacts)
├── ab_comparison/          # A/B records + figure (committed)
├── docs/                   # Captured runs: demo.gif, two_robots.gif
├── tests/                  # pytest suite (36 tests: unit + headless end-to-end)
├── .github/workflows/ci.yml# CI: runs the suite on every push / PR
├── README.md               # User-facing documentation
└── PROJECT_ARGO_SUMMARY.md # This file
```

## Tech stack

- **PyBullet 3.2.7** — physics engine and renderer, stepped at 480 Hz
  (`p.setTimeStep(1/480)`, matched to the loop's 1/480 bookkeeping so
  the GUI runs in real time)
- **Python 3.10** in `venv310`
- **Matplotlib** — evaluation learning curves, the A/B figure, and the
  live navigation map (`nav_map.py`, its own process fed over a
  non-blocking pipe so a ~60ms redraw never stalls the 480 Hz loop)
- **NumPy + Pillow** — frame snapshots and GIF encoding for
  `SIM_CAPTURE` only
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

## ideal_route (the tracked path)

```python
ideal_route = [(0,0), (14,0), (14,8), (14,12), (14,15), (14,20), (14,23)]
```

Tracked with `lookahead_point_on_route()` (pure pursuit, 1m
lookahead); never changes. A* + `smooth_path` are used only for
reactive post-collision detours.

NOT drawn in the 3D view -- the planned-route overlay (gold outbound,
cyan return) was removed there, since the road network's dashed
centrelines already show where the robot can go and the breadcrumb
trail shows where it went. The navigation map still draws both legs;
that is the view whose job is showing the plan.

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
episode, and writes `evaluation/eval_results.json` + `learning_curve.png`
(navigation time, trailing success rate, Σ|ΔQ| convergence, ε decay).

```
python evaluate.py [-n N] [--fresh] [--seed-start K] [--time-limit S] [--tag NAME]
python evaluate.py --ab N [--seed-start K] [--q-from FILE]   # paired A/B, see below
python evaluate.py --explore N [--seed-start K]              # fill never-tried actions
```

Sim-side hooks (env vars, all no-ops in normal GUI use):
`SIM_HEADLESS=1` (p.DIRECT, no sleep), `SIM_TIME_LIMIT` (abort as
failure), `SIM_SEED` (reproducibility), `SIM_CAPTURE=<path.gif>`
(offscreen GIF capture: a side-by-side composite of the 3D view via
TinyRenderer and the navigation map rendered through nav_map's own
drawing code under Agg; the trail is drawn as real disc geometry since
debug lines don't render offscreen (the route is not drawn at all,
matching the live 3D view), robot marked by its GO/STOP signal light; 8 fps, played back 2x, quantised to a run-wide shared palette).
Every run prints
`RESULT success=… sim_time=… collisions=… near_misses=… go=… wait=…
epsilon=… run=…`.

**Measured baselines (2026-07-11), the portfolio pair:**
- **Learning from scratch** — 100 unseeded episodes, blank table,
  ε 0.30→0.05 (`evaluation/learning_curve_from-scratch.png`): 100% success, mean
  25.4s, 1 collision + 6 near-misses total, with 5 of 6 incident
  episodes inside the first 21 and **zero incidents in the final 50**.
  Early episodes update the table every crossing (Σ|ΔQ| ≈ 3 plateau);
  zero-update episodes become common late.
- **Trained policy** — 100 unseeded episodes on the pre-trained table
  (`evaluation/learning_curve_trained-policy.png`): **100% success, mean 25.6s /
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

## Round trip + A/B (added 2026-08)

**A/B: learned crossing vs rule baseline** (`evaluate.py --ab N`).
`SIM_CROSSING=q|rule` swaps the service-road controller; `SIM_FREEZE_Q=1`
makes the Q arm greedy with no updates; `SIM_Q_FILE=<path>` points the
sim at a Q-table other than `q_table.json` (unreadable or wrong-layout
file = hard error, never a silent blank table). Every seed runs under both arms
(identical traffic), which removes the traffic variance that would
otherwise swamp the effect. New `hold=` field in RESULT measures seconds
held at that crossing, from the commanded signal and position, so both
policies are scored identically.

Result (40 paired seeds): **both 40/40 success, 0 collisions, 0 near
misses; mean crossing hold 0.49s (Q) vs 1.06s (rule)**. On the 23
contested seeds: 0.85s vs 1.83s, a 54% delay reduction at equal safety.
Q faster on 9 seeds, rule faster on 0, tied on 31. -> ab_comparison/ab_comparison.png
(Contested = either arm held > 0.05s; faster/tied use the same 0.05s
margin. Figures re-verified against ab_comparison.json, 2026-11.)

**Reproducing it from a clone** (2026-11): `q_table.json` is gitignored,
so plain `--ab` on a fresh clone would freeze a *blank* table. Use
`--ab 40 --q-from evaluation/eval_results_trained-policy.json`, which
hands the committed `final_q_table` to the sim via `SIM_Q_FILE`. That
table (run 121) isn't the exact one the A/B was recorded with (run 141),
but a frozen greedy policy only acts on each state's argmax. The run-141
table wasn't saved, so that can't be compared directly; the committed
table and the later run-276 `q_table.json` share every argmax, and the
re-run below made the recorded decision on every seed.

Verified by re-running exactly that with `q_table.json` removed: every
seed's success, collisions, near-misses and GO/WAIT decision matched the
recording, and both arms' mean holds came out the same (0.49s vs
1.06s). Per-seed timings drifted by ≤ 0.03s (not investigated: the sim
gained the round-trip and fleet code after the A/B was recorded, and
the Python build changed). One effect of that drift: seed 7's rule hold
went 0.05 → 0.06s, crossing the 0.05s contested threshold, so a re-run
reports 24 contested seeds / Q faster on 10 / 0.82s vs 1.76s (−53%)
instead of 23 / 9 / 0.85s vs 1.83s (−54%). Same decisions throughout;
the committed `ab_comparison.json` keeps the original recording.

**Round trip** (`SIM_RETURN=1`, opt-in). Gate -> depot by a different
route: apron (offset lane) -> Gate A connector -> service road ->
connector A. Leg state machine swaps `active_route`; crossing windows
became direction-aware and use the robot's actual x (the return crosses
the apron front at x=8, not 14).

Three geometry bugs found by measurement, all the same class -- a hold
window reaching into a NEIGHBOURING road's car lane, because the apron
roads are only 3m apart while cars occupy +/-0.35m and the robot is ~1m:
1. lane-entry standoff 1.5-3.5m parked the robot at y=18.5, inside the
   apron-back lane -> 18 collisions/episode. Narrowed to 1.2-1.8m.
2. descending crossing standoff of 1.0m left 0.15m clearance.
3. descending stop margin of 2.5m reached into the next lane, so holding
   for the apron front parked in midline traffic (9 collisions on one
   seed). Cut to 2.0m. The safe band between two roads is
   [lane_edge + robot_half, next_lane_edge - robot_half].

Also proven by measurement: a gap reservation cannot work for the 6m
apron leg (robot 1.4 m/s vs a 5 m/s car on a 20m range -> deadlock every
episode). That leg needs lane discipline, hence the y=16.8 offset.

Status: 6/6 seeds complete the round trip; 2 of 6 still take one
recoverable collision on the return leg (recovery handles it, all
succeed). Not yet at the outbound leg's 0.01/episode.

## Fleet + trails (added 2026-08)

All per-robot state (route, leg, state machine, signal light, trail,
detour bookkeeping, Q-crossing latches, metrics) lives on a `Robot`;
the loop iterates `for r in robots`, which keeps every `continue` in
the control logic meaning "skip the rest of THIS robot's frame". The
Q-table is shared, so both robots learn one policy.

Trails show only the leg in progress: arriving at a destination wipes
that robot's trail. Trail state lives in FOUR stores and all four must
be reset, which took three attempts to get right — GUI debug lines,
capture-mode disc bodies (debug lines don't render offscreen, so
recordings use discs), the live map process, and the capture panel's
own per-robot point list.

Overlap ordering: the 3D view uses fixed per-robot heights 2cm apart.
Per-segment recency ordering was tried and reverted — it needs
interleaved heights a fraction of a millimetre apart, which the depth
buffer cannot resolve at this camera distance, so overlapping trails
flickered. The 2D map has no depth buffer and orders by painter's
order, fixed per robot; raising the growing trail to the front there
caused the same flicker by swapping z-order every frame.

## Known issues / limitations

1. **Q-state has no car-speed feature** and covers only the service
   road; the apron crossings are fixed-rule. Multi-crossing learning
   would need per-road or shared state.
2. **No same-lane traffic handling** — no overtaking; the taxiway is
   kept car-free by design.
3. **Two robots, no dispatcher.** `SIM_ROBOTS=2` deploys a second robot
   when the first turns for home. They stay out of phase by
   construction — the second is outbound while the first returns — so
   they never contest the shared taxiway stretch in practice. Verified
   on seed 205: the one collision there happens after the other robot
   has already finished, i.e. it is the known return-leg lane
   fragility, not a robot-robot conflict. A third robot, or any change
   to the deployment timing, would need a real dispatcher: the natural
   fit is feeding the other robots into `lane_conflict_free()`, which
   already forward-simulates moving obstacles.
4. **No sensor simulation** — perfect state via PyBullet API.
5. **`smooth_path` can still cut corners** where short apron segments
   are geometrically close.
6. **ε floor (0.05) fills rare states slowly.** `--explore N` now
   covers this. With `SIM_EXPLORE=1`, the agent takes a state's
   never-tried action whenever it reaches that state. On the run-276
   table it filled all four gaps (0–3m, 3–5m and 5–7m receding WAIT;
   7–9m receding GO) in 19 episodes, and the greedy policy didn't
   change.
7. **WAIT values are inflated relative to GO.** The 7–9m receding
   state still prefers WAIT after GO was tried there (WAIT +75.35 vs GO
   +3.00), so lack of samples wasn't the reason. The update is
   asymmetric. A WAIT resolves while a car is still nearby, so it
   bootstraps on that state's high max (γ·max Q(s')). A GO resolves
   after the robot has cleared the road, when the next state is usually
   9m+, which is never consulted and stays 0. So one safe GO earns 0.3 ×
   10 = 3.0, while WAIT compounds towards 10/(1−γ). This is harmless in
   practice, because the predictive release crosses almost immediately.
   But it biases the learned policy towards WAIT. Treating each crossing
   as terminal (no bootstrap) for both actions would compare them
   fairly.

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
- [x] **Make the A/B reproducible from a clone** — `--ab --q-from
      evaluation/eval_results_trained-policy.json` evaluates the
      committed `final_q_table` via `SIM_Q_FILE`, leaving the gitignored
      `q_table.json` untouched (2026-11)
- [x] **Harness `--explore` flag** — `--explore N` runs learning
      episodes with `SIM_EXPLORE=1` (never-tried actions first) until
      every consulted state has tried both, then reports greedy-policy
      changes for a frozen `--ab` re-measure (2026-11)
- [ ] **Terminal crossing reward** — stop bootstrapping the
      crossing update so WAIT and GO compete on equal terms (see
      limitation 7)
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
- [x] **Tests + CI** — pytest suite (nav_map geometry, evaluate
      helpers, headless end-to-end episodes, capture pipeline) run by
      GitHub Actions on every push (.github/workflows/ci.yml).
      Still open: direct unit tests for a_star, smooth_path,
      has_clear_line, find_rejoin_target, car_time_to_crossing and
      q_state -- robot_sim.py executes at import, so those need the
      module split first (they're covered indirectly by the
      end-to-end episodes meanwhile)
- [x] **README problem framing** — *The problem* / *The result*
      sections lead with the gap-acceptance problem and the A/B table;
      `--ab` documented in the harness section (2026-11)
- [x] **Screen recording / GIF** — `docs/demo.gif` (seed 205: clean
      run with a learned WAIT at the service-road crossing), captured
      via the `SIM_CAPTURE` hook
