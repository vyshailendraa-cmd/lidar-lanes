"""Stitch consecutive KITTI raw LiDAR scans into one dense local map using GPS/IMU poses.

Single scans are sparse along the road (rings metres apart); mapping companies
accumulate many scans first. Output is in the first scan's frame (x forward).

Usage:
  python accumulate.py --base path/to/kitti_raw --date 2011_09_26 --drive 0005 \
      --start 0 --n 40 --step 2 --out data/drive0005_map.npy
Then:
  python lane_extract.py data/drive0005_map.npy --x-range 0 60
"""
import argparse
import os

import numpy as np
import pykitti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="folder containing <date>/<date>_drive_<id>_sync")
    ap.add_argument("--date", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--max-range", type=float, default=30.0, help="drop far, noisy returns per scan")
    ap.add_argument("--out", default="data/map.npy")
    args = ap.parse_args()

    frames = list(range(args.start, args.start + args.n * args.step, args.step))
    ds = pykitti.raw(args.base, args.date, args.drive, frames=frames)
    T_imu_velo = np.linalg.inv(ds.calib.T_velo_imu)

    T_w_velo = [o.T_w_imu @ T_imu_velo for o in ds.oxts]
    T_ref_inv = np.linalg.inv(T_w_velo[0])

    chunks = []
    for T, scan in zip(T_w_velo, ds.velo):
        r = np.hypot(scan[:, 0], scan[:, 1])
        scan = scan[r < args.max_range]
        xyz1 = np.c_[scan[:, :3], np.ones(len(scan))]
        p = (T_ref_inv @ T @ xyz1.T).T[:, :3]
        chunks.append(np.c_[p, scan[:, 3]])
    pts = np.vstack(chunks).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, pts)
    print(f"Accumulated {len(frames)} scans -> {len(pts):,} points -> {args.out}")


if __name__ == "__main__":
    main()
