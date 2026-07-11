"""
prepare_dataset_yolo17.py
=========================
Membangun dataset binary fall-detection dari skeleton hasil ekstraksi
YOLO11n-pose (17 joint COCO).

Desain ini IDENTIK dengan prepare_dataset_ntu25.py agar bisa
apple-to-apple comparison:
  - Format output sama  : (N, 3, 150, 17, 1)
  - Modality sama       : joint, bone, motion
  - Split sama          : stratified random 80/20 atau cross-subject (P-based)
  - Normalisasi sama    : sudah dilakukan di extract_skeleton.py
                          (hip-center, shoulder-width = 1.0)

Input:
  skeleton_dir/
    fall/          <- file .npy shape (T, 17, 3)  [x, y, conf]
    not_fall/      <- file .npy shape (T, 17, 3)
    label_map.json (opsional, dibuat otomatis jika tidak ada)

Output:
  dataset/yolo17_data/
    balanced/{joint,bone,motion}/
      train_data.npy  (N, 3, 150, 17, 1) float32
      train_label.pkl (names, labels)
      val_data.npy
      val_label.pkl
      label_map.json
      dataset_info.json
    full/{joint,bone,motion}/
      ...

Cara pakai:
    python scripts/prepare_dataset_yolo17.py \\
        --skeleton_dir dataset/ntu_skeleton \\
        --out_dir      dataset/yolo17_data \\
        --max_frames   150 \\
        --val_split    0.2
"""

SKELTON_PATH = "dataset/ntu_skeleton"
OUTPUT_PATH  = "dataset/yolo17_data"

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("pip install scikit-learn")

# ── Konstanta ──────────────────────────────────────────────────────────────────

NUM_JOINTS = 17
CHANNELS   = 3   # x, y, confidence

# Pasangan tulang COCO 17-joint untuk modality bone (child, parent) 0-indexed
# Root = joint 0 (nose), bone root = zero vector
COCO_BONE_PAIRS = [
    (1, 0),   # left_eye   → nose
    (2, 0),   # right_eye  → nose
    (3, 1),   # left_ear   → left_eye
    (4, 2),   # right_ear  → right_eye
    (5, 0),   # left_shoulder  → nose  (via leher, approx)
    (6, 0),   # right_shoulder → nose
    (7, 5),   # left_elbow  → left_shoulder
    (8, 6),   # right_elbow → right_shoulder
    (9, 7),   # left_wrist  → left_elbow
    (10, 8),  # right_wrist → right_elbow
    (11, 5),  # left_hip    → left_shoulder  (via torso)
    (12, 6),  # right_hip   → right_shoulder
    (13, 11), # left_knee   → left_hip
    (14, 12), # right_knee  → right_hip
    (15, 13), # left_ankle  → left_knee
    (16, 14), # right_ankle → right_knee
]

# Cross-subject split berdasarkan subject ID di nama file NTU
# (dipakai bila --split_method cross_subject)
XSUB60_TRAIN = {1, 2, 4, 5, 8, 9, 11, 13, 14, 15, 16, 17, 18}
XSUB60_TEST  = {3, 6, 7, 10, 12, 19, 20}

LABEL_MAP = {"not_fall": 0, "fall": 1}


# ── Helper ─────────────────────────────────────────────────────────────────────

def get_subject_id(stem: str) -> int:
    """Ekstrak subject ID dari nama file NTU: S001C001P003R001A043 → 3."""
    m = re.search(r'[Pp](\d{3})', stem)
    return int(m.group(1)) if m else -1


def count_valid_frames(seq: np.ndarray) -> int:
    """Hitung frame yang punya keypoint terdeteksi (confidence > 0)."""
    conf = seq[:, :, 2]           # (T, 17)
    valid = int((conf > 0).any(axis=1).sum())
    return max(valid, 1)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def pad_or_crop(seq: np.ndarray, target: int) -> np.ndarray:
    """
    (T, 17, 3) → (target, 17, 3).
    Crop dari tengah jika lebih panjang, pad nol di akhir jika lebih pendek.
    """
    T = seq.shape[0]
    if T == target:
        return seq
    if T < target:
        pad = np.zeros((target - T, NUM_JOINTS, CHANNELS), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)
    start = (T - target) // 2
    return seq[start: start + target]


def to_tensor(seq: np.ndarray, target: int) -> np.ndarray:
    """(T, 17, 3) → (3, target, 17, 1)"""
    s = pad_or_crop(seq, target)    # (target, 17, 3)
    t = s.transpose(2, 0, 1)        # (3, target, 17)
    return t[:, :, :, np.newaxis]   # (3, target, 17, 1)


# ── Modalities ─────────────────────────────────────────────────────────────────

def build_bone(seq: np.ndarray) -> np.ndarray:
    """
    Bone = koordinat child - koordinat parent. (T, 17, 3)
    Channel confidence = min(conf_child, conf_parent).
    Root joint (nose, idx 0) bone = nol.
    """
    b = np.zeros_like(seq)
    for child, parent in COCO_BONE_PAIRS:
        b[:, child, :2] = seq[:, child, :2] - seq[:, parent, :2]
        b[:, child, 2]  = np.minimum(seq[:, child, 2], seq[:, parent, 2])
    return b


def build_motion(seq: np.ndarray) -> np.ndarray:
    """
    Motion = perbedaan frame-ke-frame (velocity). (T, 17, 3)
    Frame terakhir = nol.
    Channel confidence = min(conf_t, conf_t+1).
    """
    m = np.zeros_like(seq)
    m[:-1, :, :2] = seq[1:, :, :2] - seq[:-1, :, :2]
    m[:-1, :, 2]  = np.minimum(seq[:-1, :, 2], seq[1:, :, 2])
    return m


# ── Pengumpulan sampel ─────────────────────────────────────────────────────────

def collect_samples(skeleton_dir: Path) -> list:
    """
    Kumpulkan semua .npy dari fall/ dan not_fall/.
    Return: list of (filepath_str, label_int, stem_name)
    """
    samples = []
    for name, idx in LABEL_MAP.items():
        folder = skeleton_dir / name
        if not folder.exists():
            print(f"  [WARN] Folder tidak ada: {folder}")
            continue
        files = sorted(folder.glob("*.npy"))
        print(f"  {name:10s} (label={idx}): {len(files)} file")
        for fp in files:
            arr = np.load(str(fp), allow_pickle=False)
            if arr.ndim == 3 and arr.shape[1] == NUM_JOINTS and arr.shape[2] >= 2:
                # Pastikan 3 channel
                if arr.shape[2] == 2:
                    # Tambah confidence = 1.0 jika belum ada
                    conf = np.ones((*arr.shape[:2], 1), dtype=np.float32)
                    arr  = np.concatenate([arr, conf], axis=2)
                samples.append((str(fp), idx, fp.stem))
            else:
                print(f"    [SKIP] shape tidak valid {arr.shape}: {fp.name}")
    return samples


# ── Build & save ───────────────────────────────────────────────────────────────

def build_tensors(samples, max_frames, transform=None):
    N      = len(samples)
    data   = np.zeros((N, CHANNELS, max_frames, NUM_JOINTS, 1), dtype=np.float32)
    labels = []
    names  = []
    errors = 0

    for i, (fp, label, stem) in enumerate(samples):
        try:
            arr = np.load(fp, allow_pickle=False).astype(np.float32)
            if arr.shape[2] == 2:
                conf = np.ones((*arr.shape[:2], 1), dtype=np.float32)
                arr  = np.concatenate([arr, conf], axis=2)
            src = transform(arr) if transform else arr
            data[i] = to_tensor(src, max_frames)
            labels.append(label)
            names.append(stem)
        except Exception as e:
            print(f"  [ERR] {fp}: {e}")
            errors += 1

        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{N}] diproses...")

    if errors:
        print(f"  [WARN] {errors} file gagal, dilewati.")
    data = data[:len(labels)]
    return data, labels, names


def save_split(out_dir: Path, split: str, data, labels, names):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / f"{split}_data.npy"), data)
    with open(out_dir / f"{split}_label.pkl", "wb") as f:
        pickle.dump((names, labels), f)
    c = Counter(labels)
    print(f"    {split}_data.npy  shape={data.shape}  "
          f"not_fall={c[0]}  fall={c[1]}")


def make_version(out_dir: Path, train_s, val_s, max_frames, version_name, transform=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{version_name}]")

    td, tl, tn = build_tensors(train_s, max_frames, transform)
    vd, vl, vn = build_tensors(val_s,   max_frames, transform)
    save_split(out_dir, "train", td, tl, tn)
    save_split(out_dir, "val",   vd, vl, vn)

    tc = Counter(tl)
    vc = Counter(vl)
    pos_weight = round(tc[0] / max(tc[1], 1), 4)

    with open(out_dir / "label_map.json", "w") as f:
        json.dump({"0": "not_fall", "1": "fall"}, f, indent=2)

    info = {
        "version":        version_name,
        "source":         "YOLO11n-pose extraction from NTU RGB+D video",
        "fall_classes":   [43],
        "not_fall_classes": [8, 9, 27, 42],
        "num_class":      2,
        "classes":        {"0": "not_fall", "1": "fall"},
        "num_joints":     NUM_JOINTS,
        "channels":       CHANNELS,
        "channel_names":  ["x_norm", "y_norm", "confidence"],
        "normalization":  "hip_center_shoulder_scale",
        "max_frames":     max_frames,
        "num_person":     1,
        "train":  {"total": len(tl), "not_fall": tc[0], "fall": tc[1]},
        "val":    {"total": len(vl), "not_fall": vc[0], "fall": vc[1]},
        "pos_weight": pos_weight,
        "class_weight_for_config": [1.0, round(pos_weight, 2)],
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"    pos_weight={pos_weight}  class_weight: [1.0, {pos_weight}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Prepare YOLO17 binary fall dataset dari skeleton .npy")
    ap.add_argument("--skeleton_dir", default="dataset/ntu_skeleton",
                    help="Direktori berisi fall/ dan not_fall/ subfolder")
    ap.add_argument("--out_dir",      default="dataset/yolo17_data")
    ap.add_argument("--max_frames",   type=int, default=150)
    ap.add_argument("--val_split",    type=float, default=0.2)
    ap.add_argument("--split_method", default="cross_subject",
                    choices=["random", "cross_subject"],
                    help="'random' = stratified 80/20 | "
                         "'cross_subject' = berdasarkan P-ID (NTU protocol)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    skel = Path(args.skeleton_dir)
    out  = Path(args.out_dir)

    print(f"\n{'='*60}")
    print("YOLO11n-pose 17-Joint Binary Fall Detection Dataset")
    print(f"{'='*60}")
    print(f"Skeleton dir : {skel}")
    print(f"Output dir   : {out}")
    print(f"Max frames   : {args.max_frames}")
    print(f"Split method : {args.split_method}")
    print(f"Modality     : joint, bone, motion")

    if not skel.exists():
        sys.exit(f"[ERROR] Direktori tidak ditemukan: {skel}")

    print("\nMengumpulkan sampel...")
    all_samples = collect_samples(skel)
    if not all_samples:
        sys.exit("[ERROR] Tidak ada sampel! Pastikan fall/ dan not_fall/ ada.")

    labels_all = [s[1] for s in all_samples]
    cnt = Counter(labels_all)
    print(f"\nTotal: {len(all_samples)}  "
          f"(not_fall={cnt[0]}, fall={cnt[1]}, "
          f"rasio 1:{cnt[0]/max(cnt[1],1):.1f})")

    # ── Split ──────────────────────────────────────────────────────────────────
    if args.split_method == "cross_subject":
        print("\nMenggunakan Cross-Subject split (NTU RGB+D 60 protocol)")
        tr_all = [(fp, lbl, stem) for fp, lbl, stem in all_samples
                  if get_subject_id(stem) in XSUB60_TRAIN]
        va_all = [(fp, lbl, stem) for fp, lbl, stem in all_samples
                  if get_subject_id(stem) in XSUB60_TEST]
        # Subjek di luar 1-20 → semua ke train (NTU120 extra, sama dengan ntu25)
        unknown = [(fp, lbl, stem) for fp, lbl, stem in all_samples
                   if get_subject_id(stem) not in XSUB60_TRAIN | XSUB60_TEST]
        if unknown:
            tr_all += unknown
            print(f"  [INFO] {len(unknown)} sampel subjek >20 dimasukkan ke train (NTU120 extra)")
    else:
        print(f"\nMenggunakan Random split (val={args.val_split})")
        tr_idx, va_idx = train_test_split(
            range(len(all_samples)), test_size=args.val_split,
            stratify=labels_all, random_state=args.seed)
        tr_all = [all_samples[i] for i in tr_idx]
        va_all = [all_samples[i] for i in va_idx]

    tc = Counter(s[1] for s in tr_all)
    vc = Counter(s[1] for s in va_all)
    print(f"Train: {len(tr_all)}  (not_fall={tc[0]}, fall={tc[1]})")
    print(f"Val  : {len(va_all)}  (not_fall={vc[0]}, fall={vc[1]})")

    mods = [("joint", None), ("bone", build_bone), ("motion", build_motion)]

    # ── Versi BALANCED ────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Versi 1: BALANCED (undersample not_fall pada train)")
    rng = np.random.default_rng(args.seed)
    fall_tr     = [s for s in tr_all if s[1] == 1]
    not_fall_tr = [s for s in tr_all if s[1] == 0]
    n_keep = len(fall_tr)
    if len(not_fall_tr) > n_keep:
        idx_u = rng.choice(len(not_fall_tr), size=n_keep, replace=False)
        not_fall_tr = [not_fall_tr[i] for i in sorted(idx_u)]
    tr_bal = fall_tr + not_fall_tr
    rng.shuffle(tr_bal)
    c_bal = Counter(s[1] for s in tr_bal)
    print(f"  Train balanced: {len(tr_bal)} "
          f"(not_fall={c_bal[0]}, fall={c_bal[1]})")

    bal = out / "balanced"
    for mod, fn in mods:
        make_version(bal / mod, tr_bal, va_all, args.max_frames,
                     f"balanced/{mod}", transform=fn)

    # ── Versi FULL ────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Versi 2: FULL (semua data + class weight)")
    full = out / "full"
    for mod, fn in mods:
        make_version(full / mod, tr_all, va_all, args.max_frames,
                     f"full/{mod}", transform=fn)

    # ── Summary ───────────────────────────────────────────────────────────────
    with open(full / "joint" / "dataset_info.json") as f:
        pw = json.load(f)["pos_weight"]

    print(f"\n{'='*60}")
    print("SELESAI!")
    print(f"Output di: {out}")
    print(f"\nUntuk config/fall-detection-yolo/full.yaml:")
    print(f"  loss_args:")
    print(f"    weight: [1.0, {pw}]")
    print(f"\nTraining (balanced, joint):")
    print(f"  python main.py --config config/fall-detection-yolo/balanced.yaml "
          f"--work-dir work_dir/fall_yolo17_balanced --device 0")
    print(f"\nTraining (full, joint):")
    print(f"  python main.py --config config/fall-detection-yolo/full.yaml "
          f"--work-dir work_dir/fall_yolo17_full --device 0")


if __name__ == "__main__":
    main()
