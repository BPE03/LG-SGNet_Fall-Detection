# prepare_yolo17_ntu60.py
#
# Converts YOLO11n-pose 17-joint _rgb.npy skeleton files into the .npz format
# expected by LG-SGNet's feeder (same structure as CTR-GCN's NTU60_CS.npz).
#
# Input folder layout (as given by your colleague):
#   skes_dir/
#     fall/        ← label 1  (A043)
#       S001C001P001R001A043_rgb.npy
#       ...
#     not_fall/    ← label 0  (A008, A009, A027, A042)
#       S001C001P001R001A008_rgb.npy
#       ...
#
# Each .npy file: shape (num_frames, 17, 3)  — x, y, confidence (float32)
#
# Outputs (written to save_dir/):
#   NTU60_CS.npz        — Cross Subject split
#   NTU60_CV.npz        — Cross View split
#   dataset_info.json   — class counts, imbalance ratio, class_weight_for_config
#
# Keys in each .npz:
#   x_train / x_test : (N, max_frames, 51)  float32
#   y_train / y_test : (N, 2)               float32  one-hot
#
# Modes:
#   --mode full      Use all sequences. Handle imbalance via loss weights
#                    (class_weight_for_config in dataset_info.json).
#   --mode balanced  Randomly subsample not_fall down to match fall count
#                    before splitting, so train and test sets are 50/50.
#
# Usage:
#   # Full dataset (imbalanced)
#   python prepare_yolo17_ntu60.py \
#       --skes_dir   /path/to/skeleton_root/ \
#       --index_file /path/to/file_index.json \
#       --save_dir   ./data/ntu/full/ \
#       --mode       full
#
#   # Balanced dataset (equal fall / not_fall)
#   python prepare_yolo17_ntu60.py \
#       --skes_dir   /path/to/skeleton_root/ \
#       --index_file /path/to/file_index.json \
#       --save_dir   ./data/ntu/balanced/ \
#       --mode       balanced \
#       --seed       42

import os
import os.path as osp
import argparse
import json
import numpy as np


# ---------------------------------------------------------------------------
# CS / CV split  (identical to seq_transformation.get_indices)
# ---------------------------------------------------------------------------

CS_TRAIN_IDS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
CS_TEST_IDS  = {3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40}
CV_TRAIN_IDS = {2, 3}
CV_TEST_ID   = 1


def get_split_mask(performers, cameras, evaluation):
    if evaluation == 'CS':
        train = np.isin(performers, list(CS_TRAIN_IDS))
        test  = np.isin(performers, list(CS_TEST_IDS))
    else:  # CV
        train = np.isin(cameras, list(CV_TRAIN_IDS))
        test  = (cameras == CV_TEST_ID)
    return train, test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_name(ske_name):
    """Extract camera and performer from NTU-style filename."""
    # e.g. S001C002P003R001A043_rgb  →  camera=2, performer=3
    return {
        'camera':    int(ske_name[5:8]),
        'performer': int(ske_name[9:12]),
    }


def load_npy(path):
    """Load a single _rgb.npy and return flattened (num_frames, 51)."""
    data = np.load(path)          # (num_frames, 17, 3)
    return data.reshape(len(data), -1).astype(np.float32)


def align_frames(skes_joints, max_frames):
    """Zero-pad all sequences to max_frames."""
    n = len(skes_joints)
    out = np.zeros((n, max_frames, 51), dtype=np.float32)
    for i, joints in enumerate(skes_joints):
        out[i, :len(joints)] = joints
    return out


def one_hot(labels, num_classes=2):
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, l in enumerate(labels):
        out[i, l] = 1
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def balance_names(names, file_index, seed):
    """
    Subsample not_fall names so the total not_fall count equals the fall count.
    Sampling is done before the train/test split so both sets end up balanced.
    """
    fall_names     = [n for n in names if file_index[n] == 1]
    not_fall_names = [n for n in names if file_index[n] == 0]
    n_fall         = len(fall_names)

    rng = np.random.default_rng(seed)
    sampled_not_fall = rng.choice(not_fall_names, size=n_fall, replace=False).tolist()

    balanced = fall_names + sampled_not_fall
    print('Balanced: kept %d fall + %d not_fall (dropped %d not_fall sequences)' % (
        n_fall, n_fall, len(not_fall_names) - n_fall))
    return balanced


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.index_file, 'r') as f:
        file_index = json.load(f)   # { "S001C001P001R001A043_rgb": 1, ... }

    names = list(file_index.keys())
    print('Total sequences in index: %d' % len(names))

    if args.mode == 'balanced':
        names = balance_names(names, file_index, args.seed)

    n = len(names)
    print('Sequences to process: %d' % n)

    skes_joints = []
    labels      = np.zeros(n, dtype=int)
    performers  = np.zeros(n, dtype=int)
    cameras     = np.zeros(n, dtype=int)
    frames_cnt  = np.zeros(n, dtype=int)
    missing     = []

    for idx, name in enumerate(names):
        label      = file_index[name]
        subfolder  = 'fall' if label == 1 else 'not_fall'
        path       = osp.join(args.skes_dir, subfolder, name + '.npy')

        if not osp.exists(path):
            print('  WARNING: missing %s' % path)
            missing.append(name)
            skes_joints.append(np.zeros((1, 51), dtype=np.float32))
            frames_cnt[idx] = 1
        else:
            joints = load_npy(path)
            skes_joints.append(joints)
            frames_cnt[idx] = len(joints)

        meta           = parse_name(name)
        labels[idx]    = label
        performers[idx] = meta['performer']
        cameras[idx]   = meta['camera']

        if (idx + 1) % 500 == 0:
            print('  Loaded %d / %d' % (idx + 1, n))

    if missing:
        print('WARNING: %d files missing, replaced with zeros:' % len(missing))
        for m in missing:
            print('    ' + m)

    max_frames = int(frames_cnt.max())
    print('Aligning to max_frames=%d ...' % max_frames)
    aligned = align_frames(skes_joints, max_frames)

    # --- Save splits ---
    for evaluation in ['CS', 'CV']:
        train_mask, test_mask = get_split_mask(performers, cameras, evaluation)
        train_x = aligned[train_mask]
        test_x  = aligned[test_mask]
        train_y = one_hot(labels[train_mask])
        test_y  = one_hot(labels[test_mask])

        out_path = osp.join(args.save_dir, 'NTU60_%s.npz' % evaluation)
        print(f"Evaluation: {evaluation}")
        print(f"  Train distribution (Class 0 vs 1): {train_y.sum(axis=0)}")
        print(f"  Test distribution (Class 0 vs 1): {test_y.sum(axis=0)}")
        np.savez(out_path, x_train=train_x, y_train=train_y,
                           x_test=test_x,   y_test=test_y)
        print('Saved %s  (train: %d, test: %d)' % (out_path,
              train_mask.sum(), test_mask.sum()))

    # --- Save dataset_info.json for reference ---
    n_fall     = int((labels == 1).sum())
    n_not_fall = int((labels == 0).sum())
    pos_weight = round(n_not_fall / n_fall, 4)
    info = {
        'mode':      args.mode,
        'total':     n,
        'fall':      n_fall,
        'not_fall':  n_not_fall,
        'pos_weight':              pos_weight,
        'class_weight_for_config': [1.0, round(pos_weight, 1)],
    }
    info_path = osp.join(args.save_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print('Dataset info saved to %s' % info_path)
    print('  fall=%d  not_fall=%d  pos_weight=%.4f  → weight: %s' % (
        n_fall, n_not_fall, pos_weight, info['class_weight_for_config']))
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skes_dir',   required=True,
                        help='Root folder containing fall/ and not_fall/ subfolders')
    parser.add_argument('--index_file', required=True,
                        help='Path to file_index.json')
    parser.add_argument('--save_dir',   default='./',
                        help='Output directory for .npz files and dataset_info.json')
    parser.add_argument('--mode',       default='full', choices=['full', 'balanced'],
                        help='"full" uses all data; "balanced" subsamples not_fall to match fall count')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed for balanced subsampling (default: 42)')
    main(parser.parse_args())