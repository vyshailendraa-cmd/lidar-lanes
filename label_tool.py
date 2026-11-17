"""Click-to-label lane lines on a LiDAR drive -> hand-made ground truth.

Shows the road from above (same frame train_classifier.py uses). Click along each
real lane line, including across the gaps of dashed lines.

Controls
  left click      add a point to the current line
  right click / u undo the last point
  n               finish this line, start the next one
  c               show / hide the bright candidate points (yellow)
  s               save
  q or close      save and quit
Use the toolbar magnifier to zoom; clicks are ignored while zoom/pan is active.

Usage:
  python label_tool.py data/drive0011.npy --out data/drive0011_gt.json
"""
import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lane_extract import load_points
from train_classifier import pipeline

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("--out", required=True)
ap.add_argument("--x-range", type=float, nargs=2, default=[0.0, 60.0])
ap.add_argument("--y-max", type=float, default=12.0)
args = ap.parse_args()

print("Loading and finding the road surface (takes ~30 s)...")
pts = load_points(args.scan)
ground, cmask, _, _ = pipeline(pts, args.x_range, args.y_max, k_mad=2.5, max_rms=0.15, min_length=8.0)
rng = np.random.default_rng(0)
g = ground[rng.choice(len(ground), min(len(ground), 400_000), replace=False)]
cand = ground[cmask]

lines = [[]]
if os.path.exists(args.out):
    with open(args.out) as f:
        lines = json.load(f)["lines"] + [[]]
    print(f"Loaded {len(lines) - 1} existing lines from {args.out}")

# Free up our keys from matplotlib's defaults (s = save dialog, q = quit, c = back).
matplotlib.rcParams["keymap.save"] = []
matplotlib.rcParams["keymap.quit"] = []
matplotlib.rcParams["keymap.back"] = ["left", "backspace"]

# Display: horizontal = -y (right of vehicle +), vertical = x (forward). Same as the other plots.
fig, ax = plt.subplots(figsize=(8, 11), facecolor="#0b1020")
ax.set_facecolor("#0b1020")
lo, hi = np.percentile(g[:, 3], [5, 99])
ax.scatter(-g[:, 1], g[:, 0], c=np.clip(g[:, 3], lo, hi), cmap="gray", s=0.3, linewidths=0)
cand_art = ax.scatter(-cand[:, 1], cand[:, 0], c="#ffd166", s=2, linewidths=0)
ax.set_aspect("equal")
ax.tick_params(colors="w")
ax.set_xlabel("lateral (m, right of vehicle +)", color="w")
ax.set_ylabel("forward (m)", color="w")
line_arts = []


def redraw():
    for a in line_arts:
        a.remove()
    line_arts.clear()
    for i, l in enumerate(lines):
        if not l:
            continue
        p = np.array(l)
        current = i == len(lines) - 1
        (a,) = ax.plot(-p[:, 1], p[:, 0], "-o", ms=4, lw=2,
                       color="#ff4d6d" if current else "#3ddc97")
        line_arts.append(a)
    done = sum(1 for l in lines[:-1] if len(l) >= 2)
    ax.set_title(f"{done} lines done | current line: {len(lines[-1])} pts   "
                 "[click add | u undo | n next | c candidates | s save | q quit]",
                 color="w", fontsize=9)
    fig.canvas.draw_idle()


def save():
    keep = [l for l in lines if len(l) >= 2]
    with open(args.out, "w") as f:
        json.dump({"scan": args.scan, "frame": "sensor: x forward, y left (metres)",
                   "lines": keep}, f, indent=1)
    print(f"Saved {len(keep)} lines to {args.out}")


def on_click(e):
    tb = fig.canvas.manager.toolbar
    if e.inaxes != ax or (tb is not None and tb.mode):
        return
    if e.button == 1:
        lines[-1].append([float(e.ydata), float(-e.xdata)])   # back to (x, y)
    elif e.button == 3 and lines[-1]:
        lines[-1].pop()
    redraw()


def on_key(e):
    if e.key == "u" and lines[-1]:
        lines[-1].pop()
    elif e.key == "n" and len(lines[-1]) >= 2:
        lines.append([])
    elif e.key == "c":
        cand_art.set_visible(not cand_art.get_visible())
    elif e.key == "s":
        save()
    elif e.key == "q":
        plt.close(fig)
        return
    redraw()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("close_event", lambda e: save())
redraw()
plt.tight_layout()
plt.show()
