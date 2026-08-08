#!/usr/bin/env python3
"""Navigation map window for the ARGO simulation.

A turn-by-turn style 2D map -- road layout, planned route, driven
trail, live traffic, and the robot as a heading arrow -- rendered in
its own window alongside PyBullet's 3D view.

This runs as a SEPARATE PROCESS, launched by robot_sim.py. That is
deliberate: a full matplotlib redraw costs ~60ms, and doing that
inline would stall the 480Hz simulation loop badly enough to break
real-time pacing. Here the sim only writes a ~300-byte JSON line per
update (microseconds) and this process renders at its own pace.

Protocol -- one JSON object per line on stdin:
  {"type": "init",  "roads": [[[x1,y1],[x2,y2]], ...],
                    "route": [[x,y], ...], "bay": [x0,y0,x1,y1],
                    "target": [x,y], "car_colors": [[r,g,b], ...]}
  {"type": "state", "t": 12.3, "robot": [x, y, heading],
                    "signal": "GO"|"STOP", "state": "DRIVING",
                    "cars": [[x,y], ...], "remaining": 12.3,
                    "speed": 1.7}

Closing stdin (the sim exiting) closes the window.
"""
import json
import math
import sys
import threading

import matplotlib
if __name__ == "__main__":
    # Interactive backend only when this file IS the map process.
    # robot_sim.py imports this module with the Agg backend already
    # selected, to render the map panel of the demo GIF offscreen.
    matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button

# Map palette: light "paper" surface with mid-grey roads, matching the
# 3D view's gold planned route and red driven trail so the two windows
# read as the same scene.
SURFACE = "#eceff1"
ROAD = "#9095a0"
ROAD_EDGE = "#eceff1"   # dashed centreline drawn over the road slabs
ROUTE = "#ffc400"
TRAIL = "#e02020"
BAY = "#2e9e4f"
INK = "#1a1a1a"
INK_SOFT = "#5a5f6a"
GO_COLOR = "#1aa64b"
STOP_COLOR = "#e02020"

ROAD_HALF_WIDTH = 2.0    # metres, matches the 3D road art
FOLLOW_SPAN = 11.0       # default metres from view centre to edge
MIN_SPAN = 4.0           # closest zoom
ZOOM_STEP = 1.35         # multiplier per +/- click
RENDER_HZ = 10   # a full redraw costs ~60ms; this process competes for
                 # CPU/GPU with the 3D view, so keep the rate modest

# Live view state, mutated by the zoom buttons. `max_span` is derived
# from the world's own extent at startup so that zooming all the way
# out shows the entire tarmac, whatever the robot's position.
_view = {"span": FOLLOW_SPAN, "max_span": 32.0}

# Shared slot: the reader thread drops the newest state here, the main
# thread renders whatever it finds. Intermediate states are discarded
# on purpose -- a map only ever needs the latest position.
_latest = {"state": None, "init": None, "closed": False}
_lock = threading.Lock()


def _reader():
    """Drain stdin continuously so the sim's pipe never fills."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        with _lock:
            if msg.get("type") == "init":
                _latest["init"] = msg
            else:
                _latest["state"] = msg
    with _lock:
        _latest["closed"] = True


def robot_marker(x, y, heading, size=0.85):
    """Arrow-head polygon pointing along the robot's heading."""
    pts = [(size, 0.0), (-size * 0.62, size * 0.58), (-size * 0.62, -size * 0.58)]
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return [(x + px * cos_h - py * sin_h, y + px * sin_h + py * cos_h)
            for px, py in pts]


def road_polygon(start, end, half_width=ROAD_HALF_WIDTH):
    """Rectangle covering a road segment, so width scales with zoom."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    ux, uy = dx / length, dy / length
    px, py = -uy * half_width, ux * half_width
    return [(start[0] + px, start[1] + py), (end[0] + px, end[1] + py),
            (end[0] - px, end[1] - py), (start[0] - px, start[1] - py)]


def build_figure(init, figsize=(5.4, 5.8), controls=True):
    """Build the map figure. `controls` adds the zoom buttons; offscreen
    capture turns them off, since a recorded GIF can't be clicked."""
    fig, ax = plt.subplots(figsize=figsize)
    if hasattr(fig.canvas, "manager") and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("ARGO - Navigation View")
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Static layer: roads, gate bay, planned route.
    # Road slabs carry no outline -- segments overlap heavily (the
    # apron is a genuinely continuous paved area), and per-segment
    # edges would draw spurious lines straight across open tarmac.
    # Legibility comes from the dashed centrelines drawn on top.
    for seg in init["roads"]:
        poly = road_polygon(seg[0], seg[1])
        if poly:
            ax.add_patch(Polygon(poly, closed=True, facecolor=ROAD,
                                 edgecolor="none", zorder=1))
    for seg in init["roads"]:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], color=ROAD_EDGE, linewidth=0.9,
                linestyle=(0, (5, 4)), zorder=2, alpha=0.85)
    bx0, by0, bx1, by1 = init["bay"]
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                           facecolor=BAY, alpha=0.30, edgecolor=BAY,
                           linewidth=1.4, zorder=2))
    route = init["route"]
    ax.plot([p[0] for p in route], [p[1] for p in route],
            color=ROUTE, linewidth=3.0, solid_capstyle="round",
            solid_joinstyle="round", zorder=3)
    tx, ty = init["target"]
    ax.plot([tx], [ty], marker="v", markersize=11, color=BAY,
            markeredgecolor="white", markeredgewidth=1.2, zorder=6)
    ax.annotate("GATE B", (tx, ty), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8,
                color=BAY, weight="bold", zorder=6)

    # Dynamic layer.
    trail, = ax.plot([], [], color=TRAIL, linewidth=2.4,
                     solid_capstyle="round", zorder=4)
    cars = []
    for rgb in init["car_colors"]:
        patch = Rectangle((0, 0), 1.6, 0.7,
                          facecolor=(rgb[0], rgb[1], rgb[2]),
                          edgecolor="white", linewidth=0.7, zorder=5)
        ax.add_patch(patch)
        cars.append(patch)
    robot = Polygon(robot_marker(0, 0, 0), closed=True, facecolor=GO_COLOR,
                    edgecolor="white", linewidth=1.3, zorder=7)
    ax.add_patch(robot)

    banner = ax.text(0.5, 0.965, "GO", transform=ax.transAxes,
                     ha="center", va="top", fontsize=15, weight="bold",
                     color="white",
                     bbox=dict(boxstyle="round,pad=0.45", facecolor=GO_COLOR,
                               edgecolor="none"), zorder=10)
    readout = ax.text(0.5, 0.032, "", transform=ax.transAxes, ha="center",
                      va="bottom", fontsize=9.5, color=INK,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                edgecolor="#d3d7de"), zorder=10)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout(pad=0.6)

    # World extent, used to cap zoom-out and to keep the camera from
    # drifting off the map once the whole thing is visible.
    xs, ys = [], []
    for seg in init["roads"]:
        for (px, py) in seg:
            xs.append(px)
            ys.append(py)
    xs += [bx0, bx1]
    ys += [by0, by1]
    world = (min(xs) - ROAD_HALF_WIDTH, min(ys) - ROAD_HALF_WIDTH,
             max(xs) + ROAD_HALF_WIDTH, max(ys) + ROAD_HALF_WIDTH)
    # Zoom-out limit: since the robot stays centred, the view must be
    # able to reach the far edge of the map even from a corner (the
    # depot at 0,0), so allow a span of the full world extent.
    _view["max_span"] = max(world[2] - world[0], world[3] - world[1])

    art = dict(trail=trail, cars=cars, robot=robot, banner=banner,
               readout=readout, world=world)
    if not controls:
        return fig, ax, art

    # Zoom controls. Button objects must stay referenced or their
    # callbacks are garbage-collected and silently stop firing.
    def zoom(factor):
        _view["span"] = max(MIN_SPAN,
                            min(_view["max_span"], _view["span"] * factor))

    ax_in = fig.add_axes([0.885, 0.895, 0.052, 0.045])
    ax_out = fig.add_axes([0.885, 0.842, 0.052, 0.045])
    btn_in = Button(ax_in, "+", color="white", hovercolor="#d7dbe2")
    btn_out = Button(ax_out, "−", color="white", hovercolor="#d7dbe2")
    for btn in (btn_in, btn_out):
        btn.label.set_fontsize(15)
        btn.label.set_color(INK)
    btn_in.on_clicked(lambda _event: zoom(1.0 / ZOOM_STEP))
    btn_out.on_clicked(lambda _event: zoom(ZOOM_STEP))

    def on_key(event):
        if event.key in ("+", "="):
            zoom(1.0 / ZOOM_STEP)
        elif event.key in ("-", "_"):
            zoom(ZOOM_STEP)
    fig.canvas.mpl_connect("key_press_event", on_key)

    art["buttons"] = (btn_in, btn_out)
    return fig, ax, art


def apply_view(ax, rx, ry):
    """Keep the robot centred in the view at the current zoom level --
    the map slides under it, like a car navigation display. Zooming
    out therefore always keeps the robot in the middle rather than
    locking to the map's centre."""
    span_x = _view["span"]
    span_y = span_x * 1.08
    ax.set_xlim(rx - span_x, rx + span_x)
    ax.set_ylim(ry - span_y, ry + span_y)


def main():
    threading.Thread(target=_reader, daemon=True).start()

    # Wait for the layout message before drawing anything.
    init = None
    while init is None:
        with _lock:
            init = _latest["init"]
            if _latest["closed"]:
                return
        plt.pause(0.05)

    plt.ion()
    fig, ax, art = build_figure(init)
    trail_x, trail_y = [], []
    last_signal = None
    fig.show()

    while plt.fignum_exists(fig.number):
        with _lock:
            state = _latest["state"]
            closed = _latest["closed"]
        if state is None:
            if closed:
                break
            plt.pause(1.0 / RENDER_HZ)
            continue

        rx, ry, heading = state["robot"]
        art["robot"].set_xy(robot_marker(rx, ry, heading))

        signal = state.get("signal", "GO")
        if signal != last_signal:
            color = GO_COLOR if signal == "GO" else STOP_COLOR
            art["robot"].set_facecolor(color)
            art["banner"].set_text(signal)
            art["banner"].get_bbox_patch().set_facecolor(color)
            last_signal = signal

        if not trail_x or math.hypot(rx - trail_x[-1], ry - trail_y[-1]) > 0.12:
            trail_x.append(rx)
            trail_y.append(ry)
            art["trail"].set_data(trail_x, trail_y)

        for patch, (cx, cy) in zip(art["cars"], state["cars"]):
            patch.set_xy((cx - 0.8, cy - 0.35))

        remaining = state.get("remaining", 0.0)
        speed = state.get("speed", 0.0)
        eta = f"{remaining / speed:.0f}s" if speed > 0.25 else "--"
        art["readout"].set_text(
            f"{remaining:5.1f} m to gate     ETA {eta}     "
            f"{speed:4.1f} m/s     {state.get('state', '')}")

        apply_view(ax, rx, ry)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # stdin closed => the simulation has ended. Hold the final
        # frame briefly so the completed route is visible, then close
        # the window, mirroring the 3D view shutting down.
        if closed:
            plt.pause(2.0)
            break

        plt.pause(1.0 / RENDER_HZ)

    plt.close("all")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
