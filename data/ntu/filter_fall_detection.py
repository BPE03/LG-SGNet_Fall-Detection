# data/ntu/filter_fall_detection.py
import numpy as np
import os

# NTU-60 labels are 1-indexed in the raw data, stored 0-indexed in .npy/.npz
# Label 43 (1-indexed) → Fall Down
# Labels 8, 9, 27, 42 (1-indexed) → Non-Fall
FALL_LABELS     = {43}
NON_FALL_LABELS = {8, 9, 27, 42}
ALL_SELECTED    = FALL_LABELS | NON_FALL_LABELS

# Cross-Subject split: performer IDs used when building NTU60_CS.npz
XSUB60_TRAIN = {1, 2, 4, 5, 8, 9, 11, 13, 14, 15, 16, 17, 18}
XSUB60_TEST  = {3, 6, 7, 10, 12, 19, 20}

DATA_DIR = "./"
SPLITS = {
    "xsub": {
        "train": ("NTU60_CS.npz", XSUB60_TRAIN),
        "test":  ("NTU60_CS.npz", XSUB60_TEST),
    },
}


def filter_and_save(input_npz, split_key, performer_ids, out_prefix):
    data = np.load(input_npz, allow_pickle=True)
    x = data[f'x_{split_key}']      # (N, ...)
    y = data[f'y_{split_key}']      # (N,)  0-indexed labels

    # If performer IDs are available per sample, enforce the split boundary.
    p_key = f'p_{split_key}'
    if p_key in data:
        mask = np.isin(data[p_key], list(performer_ids))
        x, y = x[mask], y[mask]

    print(f"[{split_key}] Total samples: {len(y)}")

    selected = np.isin(y + 1, list(ALL_SELECTED))
    x_filtered = x[selected]
    y_raw      = y[selected]
    y_binary   = np.where(np.isin(y_raw + 1, list(FALL_LABELS)), 1, 0)

    print(f"[{split_key}] Filtered: {len(y_binary)}  fall: {(y_binary == 1).sum()}  non-fall: {(y_binary == 0).sum()}")

    os.makedirs(out_prefix, exist_ok=True)
    np.save(f"{out_prefix}/{split_key}_data.npy",  x_filtered)
    np.save(f"{out_prefix}/{split_key}_label.npy", y_binary)
    print(f"Saved → {out_prefix}/{split_key}_data.npy  &  {split_key}_label.npy\n")


if __name__ == "__main__":
    for protocol, splits in SPLITS.items():
        for split_key, (npz_name, performer_ids) in splits.items():
            filter_and_save(
                os.path.join(DATA_DIR, npz_name),
                split_key,
                performer_ids,
                f"fall_detection/{protocol}",
            )

    print("Done! Filtered data saved under data/ntu/fall_detection/")
