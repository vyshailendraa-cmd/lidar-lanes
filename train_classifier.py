"""Paint-vs-not-paint classifier for bright LiDAR ground points (weak supervision).

The rule-based pipeline (lane_extract.py) labels bright ground points automatically:
  paint     = within 0.12 m of a lane line that passed the quality filter
  not_paint = more than 0.40 m from every kept lane line (rails, kerbs, gravel, debris)
Points in between are ambiguous and dropped. A gradient-boosted model then learns to
separate the two from per-point features, and is tested on a spatially held-out part
of the road and, optionally, on a different drive it never trained on.

Metrics are logged with MLflow when it is installed, so the same script runs unchanged
as an Azure Machine Learning job (Azure ML captures MLflow automatically).

Usage:
  python train_classifier.py --train data/drive0001.npy --test data/drive0011.npy --out results/ml
"""
import argparse
import json
import os

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from lane_extract import (cluster_markings, fit_ground, group_into_lanes, lane_candidates,
                          load_points, merge_fragments, under_structure)

FEATURES = ["intensity", "rel_intensity", "height", "roughness",
            "bright_density", "bright_linearity", "bright_alignment", "rail_pair_score"]
RAIL_GAUGE = 1.435


# ---------- rule-based pipeline -> automatic labels ----------
def pipeline(pts, x_range, y_max, k_mad, max_rms, min_length):
    roi = pts[(pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1]) & (np.abs(pts[:, 1]) < y_max)]
    n, d = fit_ground(roi)
    gmask = np.abs(roi[:, :3] @ n + d) < 0.15
    ground = roi[gmask]
    cmask = lane_candidates(ground, k_mad=k_mad) & ~under_structure(ground, roi[~gmask], n, d)
    raw = merge_fragments(group_into_lanes(cluster_markings(ground[cmask][:, :2])))
    lanes = [l for l in raw if l["rms_residual_m"] <= max_rms
             and l["x_range"][1] - l["x_range"][0] >= min_length]
    return ground, cmask, lanes, (n, d)


def distance_to_lanes(xy, lanes):
    """Lateral distance from each point to the nearest kept lane line (inf if none covers its x)."""
    dist = np.full(len(xy), np.inf)
    for l in lanes:
        x0, x1 = l["x_range"]
        inside = (xy[:, 0] >= x0 - 0.5) & (xy[:, 0] <= x1 + 0.5)
        dy = np.abs(xy[inside, 1] - np.polyval(l["coef"], xy[inside, 0]))
        dist[inside] = np.minimum(dist[inside], dy)
    return dist


def distance_to_polylines(xy, lines):
    """Distance from each point to the nearest hand-labelled polyline segment."""
    dist = np.full(len(xy), np.inf)
    for l in lines:
        p = np.asarray(l, float)
        for a, b in zip(p[:-1], p[1:]):
            ab = b - a
            t = np.clip(((xy - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
            dist = np.minimum(dist, np.linalg.norm(xy - (a + t[:, None] * ab), axis=1))
    return dist


# ---------- features ----------
def features(ground, cmask, plane, seed=0):
    """Per-candidate features, each with a physical reason to separate paint from rails/clutter."""
    n, d = plane
    rng = np.random.default_rng(seed)
    bright = ground[cmask]
    xy = bright[:, :2]

    # Road-surface context from a subsample of all ground points (keeps queries fast).
    sub = ground[rng.choice(len(ground), min(len(ground), 300_000), replace=False)]
    gtree = cKDTree(sub[:, :2])
    sub_h = sub[:, :3] @ n + d
    btree = cKDTree(xy)

    F = np.zeros((len(bright), len(FEATURES)), np.float32)
    F[:, 0] = bright[:, 3]                                         # raw intensity
    F[:, 2] = bright[:, :3] @ n + d                                # height above road plane (rail heads sit higher)

    for i, nb in enumerate(gtree.query_ball_point(xy, r=0.5)):
        if nb:
            F[i, 1] = bright[i, 3] - np.median(sub[nb, 3])             # brighter than local road?
            F[i, 3] = np.std(sub_h[nb])                                # gravel beds are rough, asphalt is smooth

    for i, nb in enumerate(btree.query_ball_point(xy, r=1.0)):
        F[i, 4] = len(nb)                                          # bright-point density
        if len(nb) >= 3:
            c = xy[nb] - xy[nb].mean(0)
            ev, evec = np.linalg.eigh(np.cov(c.T) + 1e-9 * np.eye(2))
            F[i, 5] = (ev[1] - ev[0]) / (ev[1] + 1e-9)                 # 1 = a clean line, 0 = a blob
            F[i, 6] = abs(evec[0, 1])                                  # 1 = runs along the road

    # Rails come in pairs exactly one gauge apart; lane lines almost never do.
    for side in (-1, 1):
        probe = xy + np.array([0.0, side * RAIL_GAUGE])
        dist, _ = btree.query(probe, k=1, distance_upper_bound=0.15)
        F[:, 7] += np.isfinite(dist)
    return F


def labelled(path, args, gt=None):
    """Labels come from the rules (weak), or from hand-drawn lines when gt is given (independent)."""
    pts = load_points(path)
    ground, cmask, lanes, plane = pipeline(pts, args.x_range, args.y_max, args.k_mad, args.max_rms, args.min_length)
    bright = ground[cmask]
    y = np.full(len(bright), -1)
    if gt is None:
        dist = distance_to_lanes(bright[:, :2], lanes)
        y[dist <= 0.12] = 1                                        # paint
        y[dist > 0.40] = 0                                         # not paint
    else:
        dist = distance_to_polylines(bright[:, :2], gt)
        y[dist <= 0.20] = 1                                        # hand clicks are less exact than fitted lines
        y[dist > 0.50] = 0
    X = features(ground, cmask, plane)
    keep = y >= 0
    return X[keep], y[keep], bright[keep], len(lanes)


def evaluate(model, X, y, name):
    p = model.predict_proba(X)[:, 1]
    report = classification_report(y, p > 0.5, target_names=["not_paint", "paint"], output_dict=True, zero_division=0)
    m = {f"{name}_roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
         f"{name}_avg_precision": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
         f"{name}_paint_precision": report["paint"]["precision"],
         f"{name}_paint_recall": report["paint"]["recall"],
         f"{name}_not_paint_recall": report["not_paint"]["recall"],
         f"{name}_n": int(len(y))}
    print(f"\n[{name}]  ROC AUC {m[f'{name}_roc_auc']:.3f} | paint precision {m[f'{name}_paint_precision']:.3f} "
          f"| paint recall {m[f'{name}_paint_recall']:.3f} | not-paint recall {m[f'{name}_not_paint_recall']:.3f} | n={len(y)}")
    return m, p


def plot(points, prob, out, title, gt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 9), facecolor="#0b1020")
    ax.set_facecolor("#0b1020")
    sc = ax.scatter(-points[:, 1], points[:, 0], c=prob, cmap="RdYlGn", vmin=0, vmax=1, s=2)
    ax.set_xlabel("lateral (m, right of vehicle +)", color="w"); ax.set_ylabel("forward (m)", color="w")
    for l in gt or []:
        p = np.asarray(l)
        ax.plot(-p[:, 1], p[:, 0], color="#4cc9f0", lw=0.8, alpha=0.8)
    ax.tick_params(colors="w"); ax.set_aspect("equal"); ax.set_title(title, color="w")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04); cb.set_label("P(lane paint)", color="w")
    cb.ax.yaxis.set_tick_params(color="w"); plt.setp(cb.ax.get_yticklabels(), color="w")
    fig.tight_layout(); fig.savefig(out, dpi=160, facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None, help="a different drive the model never trains on")
    ap.add_argument("--test-gt", default=None, help="hand-labelled lines for the test drive (from label_tool.py)")
    ap.add_argument("--out", default="results/ml")
    ap.add_argument("--x-range", type=float, nargs=2, default=[0.0, 60.0])
    ap.add_argument("--y-max", type=float, default=12.0)
    ap.add_argument("--k-mad", type=float, default=2.5)
    ap.add_argument("--max-rms", type=float, default=0.15)
    ap.add_argument("--min-length", type=float, default=8.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import mlflow
        mlflow.start_run()
        mlflow.log_params({k: str(v) for k, v in vars(args).items()})
    except Exception:
        mlflow = None

    X, y, pts, n_lanes = labelled(args.train, args)
    print(f"Train drive: {len(y)} labelled bright points ({int(y.sum())} paint, {int((1 - y).sum())} not paint), "
          f"{n_lanes} lane lines from the rules")

    # Spatial split: every 4th 5 m block of road is held out, so test points are never
    # next door to training points (a random split would leak neighbours and inflate scores).
    test_block = (np.floor(pts[:, 0] / 5).astype(int) % 4) == 3
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    model.fit(X[~test_block], y[~test_block])

    metrics, p_hold = evaluate(model, X[test_block], y[test_block], "holdout")
    imp = permutation_importance(model, X[test_block], y[test_block], scoring="roc_auc", n_repeats=5, random_state=0)
    importance = dict(sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))
    print("Feature importance (drop in ROC AUC when shuffled):")
    for k, v in importance.items():
        print(f"  {k:18s} {v:+.4f}")

    p_all = model.predict_proba(X)[:, 1]
    plot(pts, p_all, os.path.join(args.out, "train_drive_predictions.png"), "Training drive: P(lane paint)")

    if args.test:
        gt = None
        if args.test_gt:
            with open(args.test_gt) as f:
                gt = json.load(f)["lines"]
            print(f"Test labels: {len(gt)} hand-drawn lines from {args.test_gt}")
        Xt, yt, pts_t, n_rule = labelled(args.test, args, gt)
        name = "unseen_drive_human" if gt else "unseen_drive"
        if gt is None and yt.sum() == 0:
            print(f"\n[{name}] the rules found {n_rule} lane lines on the test drive, so there is no paint to score. "
                  "Label it by hand with label_tool.py and pass --test-gt.")
        m_t, p_t = evaluate(model, Xt, yt, name)
        metrics.update(m_t)
        plot(pts_t, p_t, os.path.join(args.out, "unseen_drive_predictions.png"),
             "Unseen drive: P(lane paint)" + (" | blue = hand labels" if gt else ""), gt)

    joblib.dump({"model": model, "features": FEATURES}, os.path.join(args.out, "paint_classifier.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "feature_importance": importance}, f, indent=2)

    # One tidy table for Microsoft Fabric / Power BI: one row per metric.
    with open(os.path.join(args.out, "fabric_metrics.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in metrics.items():
            f.write(f"{k},{v}\n")
        for k, v in importance.items():
            f.write(f"importance_{k},{v}\n")

    if mlflow:
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)})
        mlflow.log_metrics({f"importance_{k}": v for k, v in importance.items()})
        mlflow.log_artifacts(args.out)
        mlflow.end_run()
    print(f"\nSaved model, metrics and plots to {args.out}/")


if __name__ == "__main__":
    main()
