"""Lane-marking extraction from a single LiDAR scan.

Pipeline: ROI crop -> RANSAC ground plane -> range-adaptive intensity threshold
(lane paint is retroreflective) -> DBSCAN clustering -> group by lateral offset
-> polynomial lane fit -> bird's-eye-view render + label-free quality metrics.

Usage:
  python lane_extract.py path/to/scan.bin              # KITTI velodyne .bin (x,y,z,intensity float32)
  python lane_extract.py scan.npy --out results/
"""
import argparse
import json
import os

import numpy as np
from sklearn.cluster import DBSCAN


# ---------- I/O ----------
def load_points(path):
    if path.endswith(".bin"):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    if path.endswith(".npy"):
        return np.load(path)[:, :4]
    if path.endswith((".csv", ".txt")):
        return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)[:, :4]
    raise ValueError(f"Unsupported format: {path}")


# ---------- Ground ----------
def fit_ground(pts, iters=300, thr=0.12, seed=0):
    """RANSAC plane on the lowest points. Returns (normal, d) with n.p + d = 0, normal z>0."""
    rng = np.random.default_rng(seed)
    z = pts[:, 2]
    cand = pts[z < np.percentile(z, 30) + 0.4][:, :3]
    best, best_n = None, -1
    for _ in range(iters):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        if abs(n[2]) < 0.9:          # reject planes that aren't roughly horizontal
            continue
        d = -n @ s[0]
        cnt = np.sum(np.abs(cand @ n + d) < thr)
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    # Refine with least squares on inliers
    inl = cand[np.abs(cand @ n + d) < thr]
    A = np.c_[inl[:, 0], inl[:, 1], np.ones(len(inl))]
    a, b, c = np.linalg.lstsq(A, inl[:, 2], rcond=None)[0]   # z = ax + by + c
    n = np.array([-a, -b, 1.0]); k = np.linalg.norm(n)
    return n / k, -c / k


# ---------- Lane candidates ----------
def lane_candidates(ground, bin_m=2.0, k_mad=4.0, abs_min=0.25):
    """Range-adaptive threshold: intensity falls with range and incidence angle,
    so compare each point to the road surface at a similar range."""
    r = np.hypot(ground[:, 0], ground[:, 1])
    inten = ground[:, 3]
    mask = np.zeros(len(ground), bool)
    for lo in np.arange(0, r.max() + bin_m, bin_m):
        sel = (r >= lo) & (r < lo + bin_m)
        if sel.sum() < 30:
            continue
        med = np.median(inten[sel])
        mad = np.median(np.abs(inten[sel] - med)) + 1e-3
        mask[sel] = inten[sel] > max(med + k_mad * 1.4826 * mad, abs_min)
    return mask


def under_structure(ground, nonground, n, d, radius=0.4, min_h=0.3):
    """Flag ground points with a vertical structure (pole, kerb, car) directly above:
    bright returns at the base of poles aren't paint."""
    from scipy.spatial import cKDTree
    tall = nonground[(nonground[:, :3] @ n + d) > min_h]
    if len(tall) == 0:
        return np.zeros(len(ground), bool)
    dist, _ = cKDTree(tall[:, :2]).query(ground[:, :2], k=1, distance_upper_bound=radius)
    return np.isfinite(dist)


def cluster_markings(xy, eps=0.3, min_samples=4, min_len=0.8, min_elong=2.5, x_stretch=4.0):
    """Anisotropic DBSCAN (x compressed by x_stretch, so gaps between LiDAR rings
    along the road still chain together), then keep elongated, longitudinal clusters."""
    if len(xy) < min_samples:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy / [x_stretch, 1.0])
    segs = []
    for lab in set(labels) - {-1}:
        p = xy[labels == lab]
        c = p - p.mean(0)
        evals, evecs = np.linalg.eigh(np.cov(c.T) + 1e-9 * np.eye(2))
        major = evecs[:, 1]
        length = np.ptp(c @ major)
        elong = np.sqrt(evals[1] / max(evals[0], 1e-9))
        heading = np.degrees(np.arctan2(abs(major[1]), abs(major[0])))  # 0 = along driving direction
        if length >= min_len and elong >= min_elong and heading < 25:
            segs.append(p)
    return segs


def _coverage(xs, bin_m=0.5):
    """Fraction of 0.5 m bins along the line that contain paint."""
    bins = np.arange(xs.min(), xs.max() + bin_m, bin_m)
    hist, _ = np.histogram(xs, bins)
    return float(np.mean(hist > 0)) if len(hist) else 1.0


def _local_fit(p, window=15.0):
    """Fit the far end of a growing lane (last `window` m) for extrapolation."""
    near = p[p[:, 0] > p[:, 0].max() - window]
    if np.ptp(near[:, 0]) < 1.0:
        return np.array([0.0, np.median(near[:, 1])])
    return np.polyfit(near[:, 0], near[:, 1], 1)


def group_into_lanes(segs, lat_tol=0.5, max_gap=12.0, min_extent=4.0, deg=2):
    """Chain marking segments front-to-back: each segment joins the lane whose
    extrapolated trajectory passes closest to it (handles curves and dash gaps)."""
    if not segs:
        return []
    groups = []   # each: list of segments
    for s in sorted(segs, key=lambda p: p[:, 0].min()):
        xc, yc = np.median(s[:, 0]), np.median(s[:, 1])
        best, best_err = None, lat_tol
        for g in groups:
            gp = np.vstack(g)
            if s[:, 0].min() - gp[:, 0].max() > max_gap:
                continue
            err = abs(np.polyval(_local_fit(gp), xc) - yc)
            if err < best_err:
                best, best_err = g, err
        if best is None:
            groups.append([s])
        else:
            best.append(s)

    lanes = []
    for g in groups:
        p = np.vstack(g)
        if np.ptp(p[:, 0]) >= min_extent:
            lanes.append(make_lane(p, len(g), deg))
    return lanes


def make_lane(p, n_segments, deg=2):
    extent = np.ptp(p[:, 0])
    d = deg if extent > 15 else 1
    coef = np.polyfit(p[:, 0], p[:, 1], d)
    resid = p[:, 1] - np.polyval(coef, p[:, 0])
    cov = _coverage(p[:, 0])
    return {
        "coef": coef.tolist(),
        "x_range": [float(p[:, 0].min()), float(p[:, 0].max())],
        "lateral_offset_m": float(np.polyval(coef, 10.0)),   # y at 10 m ahead
        "n_points": int(len(p)),
        "n_segments": n_segments,
        # painted fraction of the line's length: dashes leave regular gaps
        "paint_coverage": round(float(cov), 2),
        "type_guess": "dashed" if cov < 0.7 else "solid",
        "rms_residual_m": float(np.sqrt(np.mean(resid ** 2))),
        "points": p,
    }


def merge_fragments(lanes, lat_tol=0.35, max_overlap=3.0, max_gap=15.0):
    """Join pieces of the same physical line that chaining split apart:
    they sit end-to-end (little x-overlap) and agree laterally where they meet."""
    lanes = list(lanes)
    merged = True
    while merged:
        merged = False
        for i in range(len(lanes)):
            for j in range(i + 1, len(lanes)):
                a, b = sorted([lanes[i], lanes[j]], key=lambda l: l["x_range"][0])
                overlap = a["x_range"][1] - b["x_range"][0]
                if overlap > max_overlap or -overlap > max_gap:
                    continue
                xj = (a["x_range"][1] + b["x_range"][0]) / 2   # junction
                if abs(np.polyval(a["coef"], xj) - np.polyval(b["coef"], xj)) > lat_tol:
                    continue
                p = np.vstack([a["points"], b["points"]])
                new = make_lane(p, a["n_segments"] + b["n_segments"])
                lanes = [l for k, l in enumerate(lanes) if k not in (i, j)] + [new]
                merged = True
                break
            if merged:
                break
    return lanes


# ---------- Render ----------
def render_bev(roi, ground_mask, cand_mask, lanes, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = roi[ground_mask]
    fig, ax = plt.subplots(figsize=(7, 9), facecolor="#0b1020")
    ax.set_facecolor("#0b1020")
    ax.scatter(-roi[~ground_mask, 1], roi[~ground_mask, 0], s=0.3, c="#3a4a6b", label="non-ground")
    ax.scatter(-g[:, 1], g[:, 0], s=0.5, c=g[:, 3], cmap="Blues_r", vmin=0, vmax=0.6)
    c = g[cand_mask]
    ax.scatter(-c[:, 1], c[:, 0], s=3, c="#ffd166", label="high-intensity ground")
    colors = ["#ef476f", "#06d6a0", "#118ab2", "#f78c6b", "#c77dff", "#90be6d"]
    for i, ln in enumerate(lanes):
        xs = np.linspace(*ln["x_range"], 100)
        ax.plot(-np.polyval(ln["coef"], xs), xs, lw=2.5, c=colors[i % len(colors)],
                label=f"line {i}: {ln['lateral_offset_m']:+.2f} m (left +), {ln['type_guess']}")
    ax.set_xlabel("lateral (m, right of vehicle +)", color="w"); ax.set_ylabel("forward (m)", color="w")
    ax.tick_params(colors="w"); ax.set_aspect("equal")
    ax.set_title(title, color="w")
    leg = ax.legend(loc="upper right", fontsize=7, facecolor="#1b2238", labelcolor="w", markerscale=4)
    leg.get_frame().set_edgecolor("#444")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--out", default="results")
    ap.add_argument("--x-range", type=float, nargs=2, default=[3.0, 40.0])
    ap.add_argument("--y-max", type=float, default=12.0)
    ap.add_argument("--k-mad", type=float, default=2.5,
                    help="intensity threshold strictness (2.5 tuned on KITTI drive 0001; higher = stricter)")
    ap.add_argument("--max-rms", type=float, default=0.15,
                    help="drop lines whose points scatter more than this (m) around the fit: kerbs, rails, gravel")
    ap.add_argument("--min-length", type=float, default=8.0, help="drop lines shorter than this (m)")
    args = ap.parse_args()

    pts = load_points(args.scan)
    roi = pts[(pts[:, 0] > args.x_range[0]) & (pts[:, 0] < args.x_range[1]) & (np.abs(pts[:, 1]) < args.y_max)]
    n, d = fit_ground(roi)
    gmask = np.abs(roi[:, :3] @ n + d) < 0.15
    ground = roi[gmask]
    cmask = lane_candidates(ground, k_mad=args.k_mad)
    cmask &= ~under_structure(ground, roi[~gmask], n, d)
    segs = cluster_markings(ground[cmask][:, :2])
    raw = merge_fragments(group_into_lanes(segs))
    lanes = [l for l in raw if l["rms_residual_m"] <= args.max_rms
             and l["x_range"][1] - l["x_range"][0] >= args.min_length]
    lanes.sort(key=lambda l: l["lateral_offset_m"])
    rejected = len(raw) - len(lanes)

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.scan))[0]
    spacings = np.diff([l["lateral_offset_m"] for l in lanes]).round(2).tolist()
    report = {
        "scan": args.scan,
        "points_total": int(len(pts)), "points_roi": int(len(roi)), "points_ground": int(len(ground)),
        "ground_plane": {"normal": n.round(4).tolist(), "d": round(float(d), 3)},
        "candidates": int(cmask.sum()), "marking_segments": len(segs),
        "lines_rejected_by_quality_filter": rejected,
        "lanes": [{k: v for k, v in l.items() if k != "points"} for l in lanes],
        "adjacent_line_spacing_m": spacings,   # sanity check: ~3.0–3.75 m for real lanes
    }
    with open(os.path.join(args.out, f"{name}_lanes.json"), "w") as f:
        json.dump(report, f, indent=2)
    render_bev(roi, gmask, cmask, lanes, os.path.join(args.out, f"{name}_bev.png"),
               f"{name}: {len(lanes)} lane lines")

    print(f"ROI {len(roi)} pts | ground {len(ground)} | candidates {cmask.sum()} | segments {len(segs)} "
          f"| lines kept {len(lanes)}, rejected {rejected} (rms > {args.max_rms} m or < {args.min_length} m long)")
    for i, l in enumerate(lanes):
        print(f"  lane {i}: offset {l['lateral_offset_m']:+.2f} m, x {l['x_range'][0]:.0f}-{l['x_range'][1]:.0f} m, "
              f"{l['type_guess']}, rms {l['rms_residual_m']:.3f} m")
    print(f"  adjacent spacing (m): {spacings}")
    print(f"Saved {args.out}/{name}_bev.png and {name}_lanes.json")


if __name__ == "__main__":
    main()
