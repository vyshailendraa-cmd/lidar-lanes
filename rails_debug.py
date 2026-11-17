"""Print every candidate line (kept or rejected) and the gaps between parallel pairs,
to tune rail detection on real data.

Usage: python rails_debug.py data/drive0001.npy
"""
import sys

import numpy as np

from lane_extract import (_two_ridges, cluster_markings, estimate_heading, fit_ground, group_into_lanes,
                          lane_candidates, load_points, merge_fragments, rotate_xy, under_structure)

pts = load_points(sys.argv[1])
x_range, y_max, k_mad = (0.0, 60.0), 12.0, 2.5
pts = rotate_xy(pts, -estimate_heading(pts, x_range, y_max, k_mad))
roi = pts[(pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1]) & (np.abs(pts[:, 1]) < y_max)]
n, d = fit_ground(roi)
g = np.abs(roi[:, :3] @ n + d) < 0.15
G = roi[g]
c = lane_candidates(G, k_mad=k_mad) & ~under_structure(G, roi[~g], n, d)
raw = merge_fragments(group_into_lanes(cluster_markings(G[c][:, :2])))
raw.sort(key=lambda l: l["lateral_offset_m"])

print("idx  offset   x-range     rms    resid_std  ridges")
for i, l in enumerate(raw):
    p = l["points"]
    r = p[:, 1] - np.polyval(l["coef"], p[:, 0])
    ridges = _two_ridges(r, 1.435, 0.2)
    print(f"{i:3d}  {l['lateral_offset_m']:+6.2f}  {l['x_range'][0]:4.0f}-{l['x_range'][1]:<4.0f}  "
          f"{l['rms_residual_m']:.3f}  {np.std(r):.3f}      {None if ridges is None else np.round(ridges, 2)}")

print("\nParallel pairs 1.0-2.0 m apart (mean gap, wobble, overlap):")
for i in range(len(raw)):
    for j in range(i + 1, len(raw)):
        a, b = raw[i], raw[j]
        x0 = max(a["x_range"][0], b["x_range"][0]); x1 = min(a["x_range"][1], b["x_range"][1])
        if x1 - x0 < 3:
            continue
        xs = np.linspace(x0, x1, 30)
        gap = np.abs(np.polyval(a["coef"], xs) - np.polyval(b["coef"], xs))
        if 1.0 < gap.mean() < 2.0:
            print(f"  {i}-{j}: gap {gap.mean():.2f} m, wobble {gap.std():.2f} m, overlap {x1 - x0:.0f} m")
