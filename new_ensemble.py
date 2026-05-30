import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--save-dir',
                        default='work_dir/ensemble',
                        help='Directory to save ensemble scores and fall detection results')
    parser.add_argument('--fall-class',
                        default=42,
                        type=int,
                        help='0-indexed class index for "fall down" (default: 42 for NTU RGB+D 60)')
    parser.add_argument('--fall-threshold',
                        default=0.5,
                        type=float,
                        help='Score threshold for fall detection (default: 0.5)')

    arg = parser.parse_args()

    # Create save directory
    os.makedirs(arg.save_dir, exist_ok=True)

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0

    # Storage for ensembled scores
    all_scores = []   # ensembled raw scores (N, num_class)
    all_labels = []   # ground truth labels
    sample_names = [] # sample identifiers

    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        arg.alpha = [1.8, 2.3, 0.7, 0.2]

        for i in tqdm(range(len(label))):
            l = label[i]
            name, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]

            # Save ensembled score
            all_scores.append(r)
            all_labels.append(int(l))
            sample_names.append(name)

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1

        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[i]
            name, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]

            all_scores.append(r)
            all_labels.append(int(l))
            sample_names.append(name)

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1

        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            name, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22 * arg.alpha

            all_scores.append(r)
            all_labels.append(int(l))
            sample_names.append(name)

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1

        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    # ── Save ensembled scores ──────────────────────────────────────────────────
    all_scores = np.array(all_scores)   # shape: (N, num_class)
    all_labels = np.array(all_labels)   # shape: (N,)

    score_save_path = os.path.join(arg.save_dir, 'ensemble_score.pkl')
    with open(score_save_path, 'wb') as f:
        pickle.dump({'scores': all_scores, 'labels': all_labels, 'names': sample_names}, f)
    print(f'\nEnsemble scores saved to: {score_save_path}')

    # ── Fall Detection ─────────────────────────────────────────────────────────
    FALL_CLASS = arg.fall_class
    binary_label = (all_labels == FALL_CLASS).astype(int)

    # Apply softmax per sample so scores become proper probabilities [0, 1]
    def softmax(x):
        e = np.exp(x - np.max(x))   # subtract max for numerical stability
        return e / e.sum()

    all_scores_prob = np.array([softmax(s) for s in all_scores])  # (N, num_class)

    # Option A: argmax on raw scores — softmax doesn't change argmax result
    pred_class = np.argmax(all_scores, axis=1)
    binary_pred_argmax = (pred_class == FALL_CLASS).astype(int)

    # Options B & C use softmax probabilities
    fall_scores_prob = all_scores_prob[:, FALL_CLASS]   # now in [0, 1]

    # Option B: fixed threshold on softmax fall probability
    binary_pred_thresh = (fall_scores_prob >= arg.fall_threshold).astype(int)

    # Option C: find best threshold by F1 on softmax probabilities
    thresholds = np.arange(0.05, 1.0, 0.05)
    best_thresh, best_f1 = 0, 0
    for t in thresholds:
        pred_t = (fall_scores_prob >= t).astype(int)
        tp = np.sum((pred_t == 1) & (binary_label == 1))
        fp = np.sum((pred_t == 1) & (binary_label == 0))
        fn = np.sum((pred_t == 0) & (binary_label == 1))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    best_pred = (fall_scores_prob >= best_thresh).astype(int)

    def report(label, pred, name):
        tp = np.sum((pred == 1) & (label == 1))
        fp = np.sum((pred == 1) & (label == 0))
        fn = np.sum((pred == 0) & (label == 1))
        tn = np.sum((pred == 0) & (label == 0))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = (tp + tn) / len(label)
        print(f'\n--- {name} ---')
        print(f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}')
        print(f'  Accuracy:  {acc:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall:    {recall:.4f}')
        print(f'  F1 Score:  {f1:.4f}')
        return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                'accuracy': float(acc), 'precision': float(precision),
                'recall': float(recall), 'f1': float(f1)}

    print(f'\n{"="*50}')
    print(f'Fall Detection Results  (fall class index: {FALL_CLASS})')
    print(f'Total samples: {total_num}  |  Fall samples: {binary_label.sum()}')
    print('='*50)

    res_argmax = report(binary_label, binary_pred_argmax, 'Option A: Argmax (raw scores)')
    res_thresh = report(binary_label, binary_pred_thresh, f'Option B: Softmax Threshold={arg.fall_threshold}')
    res_best   = report(binary_label, best_pred,          f'Option C: Best Softmax Threshold={best_thresh:.2f}')

    # Save fall detection results
    fall_save_path = os.path.join(arg.save_dir, 'fall_detection_results.pkl')
    with open(fall_save_path, 'wb') as f:
        pickle.dump({
            'fall_class_index': FALL_CLASS,
            'fall_scores_raw': all_scores[:, FALL_CLASS],
            'fall_scores_softmax': fall_scores_prob,
            'binary_labels': binary_label,
            'best_threshold': best_thresh,
            'best_f1': best_f1,
            'results': {
                'argmax': res_argmax,
                'threshold': res_thresh,
                'best_threshold': res_best,
            }
        }, f)
    print(f'\nFall detection results saved to: {fall_save_path}')
    print(f'\nBest threshold for fall detection: {best_thresh:.2f}  (F1={best_f1:.4f})')