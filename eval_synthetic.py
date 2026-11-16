"""Compare extracted lane lines against simulated ground truth.

Usage:
  python eval_synthetic.py --pred results/sim_lanes.json --gt data/sim_gt.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--match-m", type=float, default=0.5)
    args = ap.parse_args()

    pred = json.load(open(args.pred))["lanes"]
    gt = json.load(open(args.gt))

    used, rows = set(), []
    for g in gt:
        best, err_best = None, None
        for i, p in enumerate(pred):
            if i in used:
                continue
            x0, x1 = p["x_range"]
            xs = np.linspace(x0, x1, 50)
            gy = g["offset"] + g["curve"] * xs ** 2
            err = np.abs(np.polyval(p["coef"], xs) - gy)
            if err.mean() < args.match_m and (err_best is None or err.mean() < err_best.mean()):
                best, err_best = i, err
        if best is None:
            rows.append((g, None, None, None))
        else:
            used.add(best)
            rows.append((g, pred[best], err_best.mean(), err_best.max()))

    found = sum(r[1] is not None for r in rows)
    print(f"Lane lines found: {found}/{len(gt)}  | false positives: {len(pred) - len(used)}")
    for g, p, mean_e, max_e in rows:
        if p is None:
            print(f"  GT {g['offset']:+.2f} m ({g['type']}): MISSED")
        else:
            print(f"  GT {g['offset']:+.2f} m ({g['type']}): mean lateral err {mean_e*100:.1f} cm, "
                  f"max {max_e*100:.1f} cm, coverage {p['x_range'][0]:.0f}-{p['x_range'][1]:.0f} m, "
                  f"pred type {p['type_guess']}")


if __name__ == "__main__":
    main()
