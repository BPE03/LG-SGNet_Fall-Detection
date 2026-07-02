"""
prepare_dataset_ntu25.py
========================
Membangun dataset binary fall-detection dari file .skeleton NTU RGB+D asli.

Kelas:
  Fall    (label 1): A043 - Fall down
  Not-fall(label 0): A008 - Sitting down
                     A009 - Standing up
                     A027 - Jump up
                     A042 - Staggering

Input : direktori berisi file *.skeleton  (format NTU Kinect)
Output: dataset/ntu25_data/{balanced,full}/{joint,bone,motion}/
          train_data.npy   (N, 3, 150, 25, 1)  float32
          train_label.pkl  (names, labels)
          val_data.npy
          val_label.pkl
          dataset_info.json

Cara pakai:
    python scripts/prepare_dataset_ntu25.py \\
        --skeleton_dir /path/to/nturgbd_skeleton \\
        --out_dir      dataset/ntu25_data \\
        --max_frames   150 \\
        --split_method random        # atau 'cross_subject'
"""

SKELETON_PATH = "e:\\000 tugasakhir\\03 code\\block-gcn-yolo\\data\\nturgbd_raw\\nturgb+d_skeletons"
OUTPUT_PATH  = "dataset/ntu25_data"

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("pip install scikit-learn")

# ── Konstanta ─────────────────────────────────────────────────────────────────

NUM_JOINTS = 25
CHANNELS   = 3   # x, y, z
MAX_BODIES = 2   # NTU bisa punya 2 orang per frame
SPINE_JOINT = 20  # joint 20 (0-indexed) = SpineBase/spine tengah sebagai pusat

# Kelas yang kita gunakan (action ID dalam filename = A0XX)
FALL_CLASSES     = {43}
NOT_FALL_CLASSES = {8, 9, 27, 42}
ALL_CLASSES      = FALL_CLASSES | NOT_FALL_CLASSES

# Cross-subject train subject IDs (NTU RGB+D 60)
XSUB60_TRAIN = {1, 2, 4, 5, 8, 9, 11, 13, 14, 15, 16, 17, 18}
XSUB60_TEST  = {3, 6, 7, 10, 12, 19, 20}

# Pasangan tulang NTU 25-joint (1-indexed → 0-indexed dalam pair tuple)
# Urutan: (child, parent) dalam 0-indexed
NTU_BONE_PAIRS_0IDX = [
    (0, 1),   # SpineBase → SpineMid
    (1, 20),  # SpineMid → SpineShoulder
    (2, 20),  # Neck → SpineShoulder
    (3, 2),   # Head → Neck
    (4, 20),  # ShoulderLeft → SpineShoulder
    (5, 4),   # ElbowLeft → ShoulderLeft
    (6, 5),   # WristLeft → ElbowLeft
    (7, 6),   # HandLeft → WristLeft
    (8, 20),  # ShoulderRight → SpineShoulder
    (9, 8),   # ElbowRight → ShoulderRight
    (10, 9),  # WristRight → ElbowRight
    (11, 10), # HandRight → WristRight
    (12, 0),  # HipLeft → SpineBase
    (13, 12), # KneeLeft → HipLeft
    (14, 13), # AnkleLeft → KneeLeft
    (15, 14), # FootLeft → AnkleLeft
    (16, 0),  # HipRight → SpineBase
    (17, 16), # KneeRight → HipRight
    (18, 17), # AnkleRight → KneeRight
    (19, 18), # FootRight → AnkleRight
    (21, 6),  # HandTipLeft → WristLeft
    (22, 6),  # ThumbLeft → WristLeft
    (23, 10), # HandTipRight → WristRight
    (24, 10), # ThumbRight → WristRight
]
# Joint 20 (SpineShoulder) adalah root, bone-nya tetap nol


# ── Parser .skeleton ───────────────────────────────────────────────────────────

def read_skeleton_file(filepath: str):
    """
    Parse file .skeleton NTU RGB+D.
    Return: list of frames, setiap frame adalah list of bodies.
    Setiap body: {'tracking_state': float, 'joints': np.ndarray(25, 3)}
    """
    frames = []
    with open(filepath, 'r') as f:
        num_frames = int(f.readline().strip())
        for _ in range(num_frames):
            num_bodies = int(f.readline().strip())
            bodies = []
            for _ in range(num_bodies):
                body_line = f.readline().strip().split()
                tracking_state = float(body_line[-1]) if body_line else 0.0
                num_joints = int(f.readline().strip())
                joints = np.zeros((num_joints, 3), dtype=np.float32)
                joint_tracking = np.zeros(num_joints, dtype=np.float32)
                for j in range(num_joints):
                    vals = list(map(float, f.readline().strip().split()))
                    joints[j, 0] = vals[0]   # x (world meter)
                    joints[j, 1] = vals[1]   # y (world meter)
                    joints[j, 2] = vals[2]   # z (world meter)
                    joint_tracking[j] = vals[11] if len(vals) > 11 else 1.0
                bodies.append({
                    'tracking_state': tracking_state,
                    'joints': joints,
                    'joint_tracking': joint_tracking,
                    'mean_tracking': joint_tracking.mean(),
                })
            frames.append(bodies)
    return num_frames, frames


def select_primary_body(frames):
    """
    Pilih satu orang utama (primary performer) dari semua frame.
    Strategi: pilih body yang paling sering muncul dan punya tracking tertinggi.
    Return: np.ndarray(T, 25, 3)
    """
    T = len(frames)
    seq = np.zeros((T, NUM_JOINTS, CHANNELS), dtype=np.float32)

    # Kumpulkan body IDs yang muncul dan rata-rata tracking-nya
    body_scores = {}
    for t, frame_bodies in enumerate(frames):
        for b_idx, body in enumerate(frame_bodies):
            key = b_idx  # gunakan urutan kemunculan
            if key not in body_scores:
                body_scores[key] = []
            body_scores[key].append(body['mean_tracking'])

    # Pilih body dengan rata-rata tracking tertinggi
    if not body_scores:
        return seq

    best_body_idx = max(body_scores, key=lambda k: np.mean(body_scores[k]))

    for t, frame_bodies in enumerate(frames):
        if best_body_idx < len(frame_bodies):
            seq[t] = frame_bodies[best_body_idx]['joints']
        elif frame_bodies:
            seq[t] = frame_bodies[0]['joints']

    return seq


def count_valid_frames(seq: np.ndarray) -> int:
    """Hitung frame yang memiliki data (bukan semua nol)."""
    valid = np.any(seq.reshape(seq.shape[0], -1) != 0, axis=1).sum()
    return max(int(valid), 1)


# ── Normalisasi & transformasi ─────────────────────────────────────────────────

def normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    """
    Normalisasi relatif terhadap pusat tulang (SpineBase, joint 0).
    seq: (T, 25, 3) → (T, 25, 3)
    """
    seq = seq.copy()
    # Cari frame valid pertama untuk referensi
    valid_frames = np.any(seq.reshape(seq.shape[0], -1) != 0, axis=1)
    if not valid_frames.any():
        return seq

    # Kurangi spine center untuk setiap frame (joint 0 = SpineBase)
    spine = seq[:, 0:1, :]   # (T, 1, 3)
    seq = seq - spine
    return seq


def pad_or_crop(seq: np.ndarray, target: int) -> np.ndarray:
    """
    (T, 25, 3) → (target, 25, 3).
    Crop dari tengah jika lebih panjang, pad nol jika lebih pendek.
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
    """(T, 25, 3) → (3, target, 25, 1)"""
    s = pad_or_crop(seq, target)    # (target, 25, 3)
    t = s.transpose(2, 0, 1)        # (3, target, 25)
    return t[:, :, :, np.newaxis]   # (3, target, 25, 1)


# ── Modalities ─────────────────────────────────────────────────────────────────

def build_bone(seq: np.ndarray) -> np.ndarray:
    """
    Bone = child - parent (displacement vector). (T, 25, 3)
    Root joint (SpineShoulder, idx 20) bone = nol.
    """
    b = np.zeros_like(seq)
    for child, parent in NTU_BONE_PAIRS_0IDX:
        b[:, child, :] = seq[:, child, :] - seq[:, parent, :]
    return b


def build_motion(seq: np.ndarray) -> np.ndarray:
    """Temporal difference (velocity). (T, 25, 3)"""
    m = np.zeros_like(seq)
    m[:-1] = seq[1:] - seq[:-1]
    return m


# ── Pengumpulan sampel ─────────────────────────────────────────────────────────

def get_action_class(filename: str) -> int:
    """Ekstrak action class dari nama file. S001C001P001R001A043.skeleton → 43"""
    stem = Path(filename).stem
    try:
        a_part = [p for p in stem.split('A') if len(p) >= 3 and p[:3].isdigit()]
        return int(a_part[-1][:3])
    except (IndexError, ValueError):
        return -1


def get_subject_id(filename: str) -> int:
    """Ekstrak subject ID. S001C001P001R001A043.skeleton → 1"""
    stem = Path(filename).stem
    try:
        p_part = [p for p in stem.split('P') if len(p) >= 3 and p[:3].isdigit()]
        return int(p_part[-1][:3])
    except (IndexError, ValueError):
        return -1


def build_yolo17_whitelist(yolo17_dir: Path) -> set:
    """
    Kumpulkan stem NTU dari skeleton YOLO17 yang sudah bersih (single-person).
    File YOLO17: S001C001P001R001A043_rgb.npy → strip '_rgb' → 'S001C001P001R001A043'.
    """
    stems = set()
    for cls_folder in ("fall", "not_fall"):
        folder = yolo17_dir / cls_folder
        if not folder.exists():
            print(f"  [WARN] Folder YOLO17 tidak ada: {folder}")
            continue
        for f in folder.glob("*.npy"):
            stem = f.stem
            if stem.endswith("_rgb"):
                stem = stem[:-4]  # strip '_rgb'
            stems.add(stem)
    return stems


def max_bodies_in_file(filepath: str) -> int:
    """Scan cepat - return jumlah body maksimum di seluruh frame tanpa parse float."""
    max_b = 0
    try:
        with open(filepath, 'r') as f:
            num_frames = int(f.readline().strip())
            for _ in range(num_frames):
                num_bodies = int(f.readline().strip())
                max_b = max(max_b, num_bodies)
                for _ in range(num_bodies):
                    f.readline()  # body info line
                    num_joints = int(f.readline().strip())
                    for _ in range(num_joints):
                        f.readline()  # joint data
    except Exception:
        return 999
    return max_b


def collect_all_samples(skeleton_dir: Path, max_bodies: int = 0,
                        yolo17_whitelist: set = None) -> list:
    """
    Scan semua file .skeleton, filter kelas yang relevan.
    yolo17_whitelist: jika diberikan, hanya ambil file yang stemnya ada di whitelist
                      (lebih cepat dari max_bodies scan, karena YOLO17 sudah bersih).
    max_bodies: fallback — jika > 0 dan yolo17_whitelist None, scan per-file.
    Return: list of (filepath, label_int, stem_name)
    """
    files = sorted(skeleton_dir.rglob("*.skeleton"))
    if not files:
        print(f"  [WARN] Tidak ada file .skeleton di {skeleton_dir}")
        return []

    print(f"  Total file .skeleton ditemukan: {len(files)}")
    samples = []
    skipped_class = 0
    skipped_filter = 0
    class_count = Counter()

    for i, fp in enumerate(files):
        act = get_action_class(fp.name)
        if act in FALL_CLASSES:
            label = 1
        elif act in NOT_FALL_CLASSES:
            label = 0
        else:
            skipped_class += 1
            continue

        if yolo17_whitelist is not None:
            if fp.stem not in yolo17_whitelist:
                skipped_filter += 1
                continue
        elif max_bodies > 0:
            if max_bodies_in_file(str(fp)) > max_bodies:
                skipped_filter += 1
                continue

        class_count[act] += 1
        samples.append((str(fp), label, fp.stem))

        if (i + 1) % 1000 == 0:
            print(f"    [{i+1}/{len(files)}] di-scan...")

    filter_label = ("whitelist YOLO17" if yolo17_whitelist is not None
                    else f"multi-person >{max_bodies}" if max_bodies > 0 else "")
    print(f"  Digunakan: {len(samples)} | Dilewati (kelas lain): {skipped_class}", end="")
    if filter_label:
        print(f" | Dilewati ({filter_label}): {skipped_filter}")
    else:
        print()
    print("  Per action class:")
    for act in sorted(class_count):
        tag = "FALL" if act in FALL_CLASSES else "not-fall"
        print(f"    A{act:03d}: {class_count[act]} sampel  [{tag}]")

    return samples


def parse_sample(filepath: str, max_frames: int) -> np.ndarray:
    """Parse satu file skeleton → (T, 25, 3) → normalisasi → pad/crop."""
    try:
        _, frames = read_skeleton_file(filepath)
    except Exception as e:
        print(f"  [ERR] Gagal parse {filepath}: {e}")
        return None

    seq = select_primary_body(frames)      # (T, 25, 3)
    seq = normalize_skeleton(seq)          # relatif ke SpineBase
    return seq                             # belum di-pad, dilakukan di to_tensor


# ── Build & save ───────────────────────────────────────────────────────────────

def build_tensors(samples, max_frames, transform=None):
    N = len(samples)
    data   = np.zeros((N, CHANNELS, max_frames, NUM_JOINTS, 1), dtype=np.float32)
    labels = []
    names  = []
    errors = 0

    for i, (fp, label, stem) in enumerate(samples):
        seq = parse_sample(fp, max_frames)
        if seq is None:
            errors += 1
            continue
        src = transform(seq) if transform else seq
        data[i] = to_tensor(src, max_frames)
        labels.append(label)
        names.append(stem)

        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{N}] diproses...")

    if errors:
        print(f"  [WARN] {errors} file gagal di-parse, dilewati.")
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

    label_map = {"0": "not_fall", "1": "fall"}
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    info = {
        "version":    version_name,
        "source":     "NTU RGB+D .skeleton",
        "fall_classes":     list(sorted(FALL_CLASSES)),
        "not_fall_classes": list(sorted(NOT_FALL_CLASSES)),
        "num_class":  2,
        "classes":    label_map,
        "num_joints": NUM_JOINTS,
        "channels":   CHANNELS,
        "channel_names": ["x_world_m", "y_world_m", "z_world_m"],
        "max_frames": max_frames,
        "num_person": 1,
        "train": {"total": len(tl), "not_fall": tc[0], "fall": tc[1]},
        "val":   {"total": len(vl), "not_fall": vc[0], "fall": vc[1]},
        "pos_weight": pos_weight,
        "class_weight_for_config": [1.0, round(pos_weight, 2)],
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"    pos_weight={pos_weight} → class_weight: [1.0, {pos_weight}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Prepare NTU RGB+D binary fall dataset dari file .skeleton")
    ap.add_argument("--skeleton_dir",  default=SKELETON_PATH,
                    help="Direktori berisi file *.skeleton")
    ap.add_argument("--out_dir",       default=OUTPUT_PATH,
                    help="Direktori output dataset")
    ap.add_argument("--max_frames",    type=int, default=150,
                    help="Jumlah frame per sampel (default 150)")
    ap.add_argument("--val_split",     type=float, default=0.2,
                    help="Proporsi validasi (default 0.2)")
    ap.add_argument("--split_method",  default="cross_subject",
                    choices=["random", "cross_subject"],
                    help="Metode split: 'random' atau 'cross_subject' (NTU60 protocol)")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--max_bodies",    type=int, default=0,
                    help="Maks jumlah orang per frame (0=nonaktif, 1=single-person via scan). "
                         "Diabaikan jika --yolo17_dir diberikan.")
    ap.add_argument("--yolo17_dir",   default="dataset/ntu_skeleton",
                    help="Direktori skeleton YOLO17 (berisi fall/ dan not_fall/). "
                         "Jika ada, gunakan whitelist stem dari dataset ini (lebih cepat & akurat).")
    args = ap.parse_args()

    skel = Path(args.skeleton_dir)
    out  = Path(args.out_dir)

    print(f"\n{'='*60}")
    print("NTU RGB+D 25-Joint Binary Fall Detection Dataset")
    print(f"{'='*60}")
    print(f"Skeleton dir : {skel}")
    print(f"Output dir   : {out}")
    print(f"Max frames   : {args.max_frames}")
    print(f"Split method : {args.split_method}")
    yolo17_dir = Path(args.yolo17_dir)
    print(f"Kelas FALL   : {sorted(FALL_CLASSES)}")
    print(f"Kelas not-fall: {sorted(NOT_FALL_CLASSES)}")

    # Tentukan strategi filter
    whitelist = None
    if yolo17_dir.exists():
        print(f"\nMembangun whitelist dari YOLO17 dir: {yolo17_dir}")
        whitelist = build_yolo17_whitelist(yolo17_dir)
        print(f"  {len(whitelist)} file unik dalam whitelist YOLO17")
        print(f"Filter       : whitelist YOLO17 (single-person, sudah bersih)")
    elif args.max_bodies > 0:
        print(f"Filter       : single-person via scan (max_bodies={args.max_bodies})")
    else:
        print(f"Filter       : nonaktif (semua file diproses)")
    print()

    if not skel.exists():
        sys.exit(f"[ERROR] Direktori tidak ditemukan: {skel}")

    print("Mengumpulkan sampel...")
    all_samples = collect_all_samples(skel, max_bodies=args.max_bodies,
                                      yolo17_whitelist=whitelist)
    if not all_samples:
        sys.exit("[ERROR] Tidak ada sampel yang ditemukan!")

    labels_all = [s[1] for s in all_samples]
    cnt = Counter(labels_all)
    print(f"\nTotal sampel: {len(all_samples)}  "
          f"(not_fall={cnt[0]}, fall={cnt[1]}, "
          f"rasio 1:{cnt[0]/max(cnt[1],1):.1f})")

    # ── Train/Val split ──────────────────────────────────────────────────────
    if args.split_method == "cross_subject":
        print("\nMenggunakan Cross-Subject split (NTU RGB+D 60 protocol)")
        tr_all = [(fp, lbl, stem) for fp, lbl, stem in all_samples
                  if get_subject_id(stem) in XSUB60_TRAIN]
        va_all = [(fp, lbl, stem) for fp, lbl, stem in all_samples
                  if get_subject_id(stem) in XSUB60_TEST]
        # File dengan subject ID di luar 1-20 → masuk train (default)
        unknown = [(fp, lbl, stem) for fp, lbl, stem in all_samples
                   if get_subject_id(stem) not in XSUB60_TRAIN | XSUB60_TEST]
        if unknown:
            print(f"  [INFO] {len(unknown)} file dengan subject > 20, "
                  "dimasukkan ke train (NTU120 extra)")
            tr_all += unknown
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
    print(f"  Train balanced: {len(tr_bal)} "
          f"(not_fall={Counter(s[1] for s in tr_bal)[0]}, "
          f"fall={Counter(s[1] for s in tr_bal)[1]})")

    bal = out / "balanced"
    mods = [("joint", None), ("bone", build_bone), ("motion", build_motion)]
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
    print(f"\nUntuk config/fall-detection-ntu/full.yaml:")
    print(f"  loss_args:")
    print(f"    weight: [1.0, {pw}]")
    print(f"\nTraining (balanced):")
    print(f"  python main.py --config config/fall-detection-ntu/balanced.yaml "
          "--work-dir work_dir/fall_ntu25_balanced --device 0")
    print(f"\nTraining (full):")
    print(f"  python main.py --config config/fall-detection-ntu/full.yaml "
          "--work-dir work_dir/fall_ntu25_full --device 0")


if __name__ == "__main__":
    main()
