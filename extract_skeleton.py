"""
step1_extract_ntu_skeleton.py
==============================
Scan satu folder flat berisi semua .avi NTU RGB+D,
filter A043 (fall) dan A008/A009/A027/A042 (not_fall),
ekstrak 17 keypoint YOLO11n-pose per frame,
normalisasi posisi, simpan .npy + index nama file.

Input folder:
    nturgb+d_rgb/
        S001C001P001R001A008.avi
        S001C001P001R001A043.avi
        ...  (semua campur)

Output:
    skeleton_raw/
        fall/
            S001C001P001R001A043.npy   shape=(T, 17, 3)
        not_fall/
            S001C001P001R001A008.npy
        label_map.json
        file_index.json       ← PENTING: {filename: label} untuk test script
        extraction_stats.json

Cara pakai:
    python step1_extract_ntu_skeleton.py \
        --ntu_dir   /path/ke/nturgb+d_rgb \
        --out_dir   skeleton_raw \
        --max_frames 150

    # Resume jika terhenti:
    python step1_extract_ntu_skeleton.py ... --resume
"""

import argparse
import json
import re
import sys   
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Install: pip install ultralytics opencv-python")

# ── Konfigurasi kelas ──────────────────────────────────────────────────────────
FALL_CLASSES     = {43}
NOT_FALL_CLASSES = {8, 9, 27, 42}
ALL_CLASSES      = FALL_CLASSES | NOT_FALL_CLASSES

CLASS_NAMES = {8: "sit down", 9: "stand up", 27: "jump up",
               42: "stagger",  43: "fall down"}

LABEL_MAP  = {"not_fall": 0, "fall": 1}
NUM_JOINTS = 17
VIDEO_EXT  = {".avi", ".mp4", ".mov", ".mkv"}


def parse_action_class(stem: str) -> Optional[int]:
    m = re.search(r'[Aa](\d{3})', stem)
    return int(m.group(1)) if m else None


def label_name(cls: int) -> str:
    return "fall" if cls in FALL_CLASSES else "not_fall"


def normalize_skeleton(sk: np.ndarray):
    """
    Translasi ke hip-center, scale ke jarak bahu kiri-kanan = 1.
    Input : (T, 17, 3)
    Output: (sk_norm, sc, hc)
        sk_norm : (T, 17, 3) ternormalisasi
        sc      : float — shoulder scale (pixel)
        hc      : (T, 2) — hip-center per frame (pixel)
    """
    sk = sk.copy()
    xy = sk[:, :, :2]
    hc = (xy[:, 11] + xy[:, 12]) / 2.0          # (T, 2)
    xy -= hc[:, np.newaxis, :]
    d  = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)
    sc = float(d[d > 1e-5].mean()) if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        xy /= sc
    sk[:, :, :2] = xy
    return sk, sc, hc


def denormalize_skeleton(sk_norm: np.ndarray, sc: float, hc: np.ndarray) -> np.ndarray:
    """Kebalikan normalize_skeleton — kembalikan ke koordinat piksel asli."""
    sk = sk_norm.copy()
    sk[:, :, :2] = sk[:, :, :2] * sc + hc[:, np.newaxis, :]
    return sk


def extract_video(model, path: str, max_frames: int):
    """
    Return: (sk_norm, sc, hc) atau None jika gagal.
        sk_norm : (T, 17, 3) ternormalisasi
        sc      : float
        hc      : (T, 2)
    """
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    take = set(map(int, np.linspace(0, total - 1, min(total, max_frames))))
    seq, bad, fi = [], 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in take:
            kf = np.zeros((NUM_JOINTS, 3), np.float32)
            r  = model(frame, verbose=False)
            if r and r[0].keypoints is not None and len(r[0].keypoints) > 0:
                kps  = r[0].keypoints
                conf = kps.conf.cpu().numpy()
                best = int(conf.mean(axis=1).argmax())
                kf[:, :2] = kps.xy.cpu().numpy()[best]
                kf[:, 2]  = conf[best]
            else:
                bad += 1
            seq.append(kf)
        fi += 1

    cap.release()
    if not seq or bad / len(seq) > 0.6:
        return None
    return normalize_skeleton(np.stack(seq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntu_dir",     default="../dataset/ntu_videos")
    ap.add_argument("--out_dir",     default="../dataset/ntu_skeleton")
    ap.add_argument("--model",       default="yolo11n-pose.pt")
    ap.add_argument("--max_frames",  type=int, default=150)
    ap.add_argument("--resume",      action="store_true")
    args = ap.parse_args()

    ntu = Path(args.ntu_dir)
    out = Path(args.out_dir)
    (out / "fall").mkdir(parents=True, exist_ok=True)
    (out / "not_fall").mkdir(parents=True, exist_ok=True)

    with open(out / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    # Scan file
    videos = [(p, parse_action_class(p.stem))
              for p in sorted(ntu.iterdir())
              if p.suffix.lower() in VIDEO_EXT]
    videos = [(p, c) for p, c in videos if c in ALL_CLASSES]

    if not videos:
        sys.exit(f"Tidak ada video NTU ditemukan di {ntu}\n"
                 f"Pastikan nama file mengandung Axxx, misal: S001C001P001R001A043.avi")

    per_cls = defaultdict(int)
    for _, c in videos:
        per_cls[c] += 1

    print(f"\nVideo ditemukan:")
    for c in sorted(ALL_CLASSES):
        lbl = "fall" if c in FALL_CLASSES else "not_fall"
        print(f"  A{c:03d} {CLASS_NAMES[c]:12s} → {lbl:10s}  {per_cls[c]} video")
    print(f"\n  Total fall    : {sum(per_cls[c] for c in FALL_CLASSES)}")
    print(f"  Total not_fall: {sum(per_cls[c] for c in NOT_FALL_CLASSES)}")

    print(f"\nMemuat YOLO: {args.model}")
    model = YOLO(args.model)

    stats      = {"success": 0, "failed": 0, "skipped": 0}
    file_index = {}   # {stem: label_int}  — untuk test script

    for vp, cls in videos:
        lname    = label_name(cls)
        out_path = out / lname / f"{vp.stem}.npy"

        if args.resume and out_path.exists():
            stats["skipped"] += 1
            stats["success"] += 1
            file_index[vp.stem] = LABEL_MAP[lname]
            continue

        print(f"  [{cls:03d}→{lname}] {vp.name} ... ", end="", flush=True)
        result = extract_video(model, str(vp), args.max_frames)

        if result is None:
            print("GAGAL")
            stats["failed"] += 1
            continue

        arr, sc, hc = result
        np.save(str(out_path), arr)
        # Simpan metadata normalisasi — dibutuhkan untuk reverse (visualisasi, debug)
        np.savez(str(out_path).replace(".npy", "_meta.npz"),
                 sc=np.float32(sc), hc=hc.astype(np.float32))
        file_index[vp.stem] = LABEL_MAP[lname]
        stats["success"] += 1
        print(f"OK shape={arr.shape}  sc={sc:.1f}")

    # Simpan index nama file → label (PENTING untuk test script)
    with open(out / "file_index.json", "w") as f:
        json.dump(file_index, f, indent=2)

    with open(out / "extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Selesai: {stats['success']} berhasil, {stats['failed']} gagal")
    print(f"file_index.json tersimpan → dibutuhkan oleh test script")


if __name__ == "__main__":
    main()