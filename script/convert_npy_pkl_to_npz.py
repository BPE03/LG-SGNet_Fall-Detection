"""
convert_npy_pkl_to_npz.py
=========================
Converts the npy + pkl data produced by prepare_dataset_yolo17.py into the
NTU60_CS.npz format expected by feeders.feeder_ntu_yolo.Feeder.

Input layout (data/ntuYOLO1 or similar):
  <input_dir>/
    balanced/
      joint/  bone/  motion/
        train_data.npy   (N, 3, T, 17, 1)  float32
        train_label.pkl  (names, labels)    pickle
        val_data.npy
        val_label.pkl
    full/
      joint/  bone/  motion/
        ...

Output NPZ keys:
  x_train / x_test : (N, T, 51)   float32
  y_train / y_test : (N, 2)        float32  one-hot

Usage:
  python convert_npy_pkl_to_npz.py
  python convert_npy_pkl_to_npz.py --input_dir data/ntuYOLO1 --output_dir data/ntuYOLO1
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_data(npy_path: Path) -> np.ndarray:
    """Load (N, 3, T, 17, 1) → reshape to (N, T, 51)."""
    arr = np.load(str(npy_path))          # (N, 3, T, 17, 1)
    N, C, T, V, M = arr.shape
    arr = arr.squeeze(-1)                  # (N, 3, T, 17)
    arr = arr.transpose(0, 2, 3, 1)        # (N, T, 17, 3)
    arr = arr.reshape(N, T, V * C)         # (N, T, 51)
    return arr.astype(np.float32)


def load_labels(pkl_path: Path, n: int) -> np.ndarray:
    """Load pkl → (names, labels) or plain list → one-hot (N, 2)."""
    with open(str(pkl_path), "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, (tuple, list)) and len(raw) == 2 and not isinstance(raw[0], int):
        labels = list(raw[1])
    else:
        labels = list(raw)

    assert len(labels) == n, f"Label count {len(labels)} != data count {n}"
    out = np.zeros((n, 2), dtype=np.float32)
    for i, lbl in enumerate(labels):
        out[i, int(lbl)] = 1.0
    return out


def convert_dir(subdir: Path, out_dir: Path, npz_name: str) -> None:
    train_npy = subdir / "train_data.npy"
    train_pkl = subdir / "train_label.pkl"
    val_npy   = subdir / "val_data.npy"
    val_pkl   = subdir / "val_label.pkl"

    for p in (train_npy, train_pkl, val_npy, val_pkl):
        if not p.exists():
            print(f"  [SKIP] missing {p}")
            return

    print(f"  Converting {subdir} ...")
    x_train = load_data(train_npy)
    y_train = load_labels(train_pkl, len(x_train))
    x_test  = load_data(val_npy)
    y_test  = load_labels(val_pkl, len(x_test))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / npz_name
    np.savez(str(out_path),
             x_train=x_train, y_train=y_train,
             x_test=x_test,   y_test=y_test)

    print(f"  Saved  {out_path}")
    print(f"    x_train {x_train.shape}  y_train {y_train.shape}"
          f"  (class 0={int(y_train[:,0].sum())}  class 1={int(y_train[:,1].sum())})")
    print(f"    x_test  {x_test.shape}   y_test  {y_test.shape}"
          f"  (class 0={int(y_test[:,0].sum())}   class 1={int(y_test[:,1].sum())})")


def main():
    ap = argparse.ArgumentParser(
        description="Convert npy+pkl dataset to NTU60_CS.npz format")
    ap.add_argument("--input_dir",  default="data/ntu25_data",
                    help="Root directory of the npy/pkl dataset")
    ap.add_argument("--output_dir", default=None,
                    help="Root for output .npz files. "
                         "Defaults to --input_dir (saves alongside npy/pkl)")
    ap.add_argument("--npz_name",   default="dataset.npz",
                    help="Output filename (default: dataset.npz)")
    args = ap.parse_args()

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir) if args.output_dir else input_root

    if not input_root.exists():
        raise SystemExit(f"[ERROR] input_dir not found: {input_root}")

    modalities = ["joint", "bone", "motion"]
    splits     = ["balanced", "full"]

    converted = 0
    for split in splits:
        for mod in modalities:
            subdir  = input_root  / split / mod
            out_dir = output_root / split / mod
            if not subdir.exists():
                print(f"  [SKIP] {subdir} does not exist")
                continue
            convert_dir(subdir, out_dir, args.npz_name)
            converted += 1

    print(f"\nDone. {converted} NPZ file(s) written.")


if __name__ == "__main__":
    main()
