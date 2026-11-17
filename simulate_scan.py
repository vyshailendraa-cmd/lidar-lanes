"""Simulate a 64-beam LiDAR scan of a 3-lane road with known lane-line positions.

Used as a ground-truth test: lane_extract.py should recover the lines, and
eval_synthetic.py reports lateral error. Includes range/incidence intensity falloff,
sensor noise, a slight curve, cars, and roadside clutter.

Usage:
  python simulate_scan.py --out data/sim.npy [--accumulate 20] [--curve 0.004]
"""
import argparse
import json
import os

import numpy as np

LANE_W = 3.5
SENSOR_H = 1.73
DASH_ON, DASH_OFF = 3.0, 6.0


def lane_lines(curve):
    # (offset at x=0, type). Ego in middle lane.
    lines = [(-1.5 * LANE_W, "solid"), (-0.5 * LANE_W, "dashed"),
             (0.5 * LANE_W, "dashed"), (1.5 * LANE_W, "solid")]
    return [{"offset": o, "type": t, "curve": curve} for o, t in lines]


def road_y(x, offset, curve):
    return offset + curve * x ** 2


def scan(ego_x, lines, rng, n_beams=64, n_az=2000, rails=False):
    pts = []
    elev = np.radians(np.linspace(-24.8, 2.0, n_beams))
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    E, A = np.meshgrid(elev, az, indexing="ij")
    E, A = E.ravel(), A.ravel()
    down = E < -0.01
    r = SENSOR_H / np.tan(-E[down])            # ground hit range
    x = r * np.cos(A[down]); y = r * np.sin(A[down])
    keep = (r < 60)
    x, y, r = x[keep], y[keep], r[keep]
    z = -SENSOR_H + rng.normal(0, 0.02, len(x))
    inc = np.arctan(r / SENSOR_H)
    base = 0.08 + 0.04 * rng.random(len(x))
    paint = np.zeros(len(x), bool)
    for ln in lines:
        d = np.abs(y - road_y(x + ego_x, ln["offset"], ln["curve"]))
        on = d < 0.075                              # 15 cm wide paint
        if ln["type"] == "dashed":
            on &= ((x + ego_x) % (DASH_ON + DASH_OFF)) < DASH_ON
        paint |= on
    refl = np.where(paint, 0.75, base)
    inten = refl * np.cos(inc * 0.6) * np.exp(-r / 45) + rng.normal(0, 0.015, len(x))
    # Off-road (beyond outer lines + shoulder) becomes grass/kerb: rougher, darker
    off = np.abs(y - road_y(x + ego_x, 0, lines[0]["curve"])) > 2 * LANE_W + 1
    z[off] += rng.normal(0.1, 0.08, off.sum())
    pts.append(np.c_[x, y, z, np.clip(inten, 0, 1)])

    curve = lines[0]["curve"]
    if rails:
        # Tram track beside the road: two bright steel rails 1.435 m apart on a rough gravel bed.
        # Rails are as reflective as paint, which is exactly what fools intensity-only rules.
        centre = -(2 * LANE_W + 2.5)
        bed = np.abs(y - road_y(x + ego_x, centre, curve)) < 1.3
        z[bed] = -SENSOR_H + np.abs(rng.normal(0.03, 0.03, bed.sum()))  # gravel bed: slightly rough
        inten[bed] = np.clip(0.25 + 0.15 * rng.random(bed.sum()), 0, 1) * np.exp(-r[bed] / 45)
        for off in (-0.7175, 0.7175):
            on = np.abs(y - road_y(x + ego_x, centre + off, curve)) < 0.06
            z[on] += 0.05                                            # rail head sits just above the bed
            inten[on] = 0.8 * np.cos(inc[on] * 0.6) * np.exp(-r[on] / 45) + rng.normal(0, 0.015, on.sum())
        pts[0] = np.c_[x, y, z, np.clip(inten, 0, 1)]

    # Stopped cars (world-fixed, centred in lanes) and roadside poles: non-ground clutter
    for wx, lane_y in [(14, LANE_W), (30, -LANE_W), (48, 0.0)]:
        cx, cy = wx - ego_x, road_y(wx, lane_y, curve)
        if not (-5 < cx < 60):
            continue
        m = 400
        pts.append(np.c_[cx + rng.uniform(-2, 2, m), cy + rng.uniform(-0.9, 0.9, m),
                         rng.uniform(-1.6, -0.2, m), rng.uniform(0.05, 0.9, m)])
    for wx in range(5, 80, 12):
        cx = wx - ego_x
        if not (-5 < cx < 60):
            continue
        m = 60
        pts.append(np.c_[cx + rng.normal(0, .05, m), road_y(wx, 2 * LANE_W + 2, curve) + rng.normal(0, .05, m),
                         rng.uniform(-1.7, 3, m), rng.uniform(0.2, 0.5, m)])
    return np.vstack(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sim.npy")
    ap.add_argument("--accumulate", type=int, default=1, help="number of scans to stitch (ego moves 1 m/scan)")
    ap.add_argument("--curve", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rails", action="store_true", help="add a tram track beside the road")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lines = lane_lines(args.curve)
    chunks = []
    for k in range(args.accumulate):
        s = scan(float(k), lines, rng, rails=args.rails)
        s[:, 0] += k                 # ego moved k metres; express in first-scan frame
        chunks.append(s)
    pts = np.vstack(chunks).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, pts)
    with open(args.out.replace(".npy", "_gt.json"), "w") as f:
        json.dump(lines, f)
    print(f"{len(pts):,} points ({args.accumulate} scan(s)) -> {args.out}")


if __name__ == "__main__":
    main()
