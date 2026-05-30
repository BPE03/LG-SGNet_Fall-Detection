"""
Real-time Fall Detection using LG-SGNet (4-modality ensemble)
=============================================================
Requirements:
    pip install torch opencv-python ultralytics numpy

Usage:
    python realtime_fall_detection.py \
        --joint-model      work_dir/ntu60/joint/best_model.pt \
        --bone-model       work_dir/ntu60/bone/best_model.pt \
        --joint-motion-model work_dir/ntu60/joint_motion/best_model.pt \
        --bone-motion-model  work_dir/ntu60/bone_motion/best_model.pt \
        --source 0

    --source can be:
        0           → default webcam
        1, 2, ...   → other camera index
        "rtsp://..."  → RTSP IP camera stream
        "video.mp4"   → video file
"""

import argparse
import os
import sys
import time
import threading
from collections import deque

import cv2
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn.functional as F

# ── NTU RGB+D joint definitions ────────────────────────────────────────────────
# 25 joints (0-indexed). Mapping from MediaPipe 33-landmark pose to NTU 25-joint.
# MediaPipe landmark indices → NTU joint index
# YOLO11-pose outputs 17 COCO keypoints (0-indexed)
COCO_TO_NTU = {
    0:  3,   # nose          → head
    5:  4,   # left shoulder → left shoulder
    6:  8,   # right shoulder→ right shoulder
    7:  5,   # left elbow    → left elbow
    8:  9,   # right elbow   → right elbow
    9:  6,   # left wrist    → left wrist
    10: 10,  # right wrist   → right wrist
    11: 12,  # left hip      → left hip
    12: 16,  # right hip     → right hip
    13: 13,  # left knee     → left knee
    14: 17,  # right knee    → right knee
    15: 14,  # left ankle    → left ankle
    16: 18,  # right ankle   → right ankle
    # COCO has no ear/hand/foot tip joints → derived below
}

# NTU RGB+D 60 bone pairs (parent → child, 0-indexed)
NTU_BONE_PAIRS = [
    (0, 1), (1, 20), (2, 20), (3, 2),   # spine chain
    (4, 20), (5, 4), (6, 5), (7, 6),    # left arm
    (8, 20), (9, 8), (10, 9), (11, 10), # right arm
    (12, 0), (13, 12), (14, 13), (15, 14), # left leg
    (16, 0), (17, 16), (18, 17), (19, 18), # right leg
    (21, 7), (22, 6),                    # left hand
    (23, 11), (24, 10),                  # right hand
]

NUM_JOINTS  = 25
NUM_COORDS  = 3   # x, y, z (MediaPipe gives depth z estimate)
WINDOW_SIZE = 64  # frames expected by LG-SGNet
FALL_CLASSES = {41, 42}  # 41=staggering, 42=fall down (both trigger alert)
FALL_CLASS   = 42        # primary class index used for score display

NTU60_CLASSES = {
    0: "drink water", 1: "eat meal", 2: "brush teeth", 3: "brush hair",
    4: "drop", 5: "pick up", 6: "throw", 7: "sit down", 8: "stand up",
    9: "clapping", 10: "reading", 11: "writing", 12: "tear up paper",
    13: "put on jacket", 14: "take off jacket", 15: "put on a shoe",
    16: "take off a shoe", 17: "put on glasses", 18: "take off glasses",
    19: "put on a hat/cap", 20: "take off a hat/cap", 21: "cheer up",
    22: "hand waving", 23: "kicking something", 24: "reach into pocket",
    25: "hopping", 26: "jump up", 27: "phone call", 28: "play with phone",
    29: "type on keyboard", 30: "point to something", 31: "taking a selfie",
    32: "check time", 33: "rub two hands", 34: "nod head/bow",
    35: "shake head", 36: "wipe face", 37: "salute", 38: "put palms together",
    39: "cross hands in front", 40: "sneeze/cough", 41: "staggering",
    42: "fall down", 43: "touch head", 44: "touch chest", 45: "touch back",
    46: "touch neck", 47: "nausea/vomiting", 48: "fan self",
    49: "punch/slap", 50: "kicking", 51: "pushing", 52: "pat on back",
    53: "point finger", 54: "hugging", 55: "giving object",
    56: "touch pocket", 57: "shaking hands", 58: "walking towards",
    59: "walking apart",
}
ALPHA       = [1.8, 2.3, 0.7, 0.2]  # ensemble weights: joint, bone, jmotion, bmotion


# ── Helper: import Model from LG-SGNet repo ────────────────────────────────────
def load_model(checkpoint_path: str, num_class: int = 60, device: torch.device = None):
    """
    Dynamically imports Model from the local LG-SGNet repo and loads weights.
    Assumes this script is run from the root of the LG-SGNet directory.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # LG-SGNet model is in model/LG_SGNet.py (or similar)
        import importlib
        model_module = importlib.import_module('model.LG-SGNet')
        model = model_module.Model(num_class=num_class, num_point=25, num_person=2,
                                   graph='graph.ntu_rgb_d.Graph',
                                   graph_args={'labeling_mode': 'spatial'})
    except Exception as e:
        print(f"[ERROR] Could not import LG-SGNet Model: {e}")
        print("Make sure you run this script from the root of the LG-SGNet repo.")
        sys.exit(1)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle full training checkpoints saved with model_state_dict key
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    # Handle other common checkpoint formats
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # Strip 'module.' prefix if trained with DataParallel
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Preprocessing ──────────────────────────────────────────────────────────────
def coco_to_ntu_skeleton(keypoints: np.ndarray) -> np.ndarray:
    """
    Convert YOLO11-pose 17 COCO keypoints → NTU 25-joint array (x, y, z).

    keypoints : np.ndarray of shape (17, 3) — (x_px, y_px, confidence)
                in image pixel coordinates from YOLO.
    Returns   : np.ndarray of shape (25, 3) centered on spine base.

    YOLO gives pixel coords, not metric 3D. We normalize to [-1, 1] using
    the skeleton's own bounding box so the scale is body-relative, which
    is consistent with how NTU skeletons look after seq_transformation.py.
    """
    joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)

    # Map COCO → NTU joints. Only use detections with confidence > 0.3.
    for coco_idx, ntu_idx in COCO_TO_NTU.items():
        x_px, y_px, conf = keypoints[coco_idx]
        if conf > 0.3:
            # x=right, flip y so up is positive (image y goes downward)
            joints[ntu_idx] = [x_px, -y_px, 0.0]

    # ── Derive missing NTU joints ────────────────────────────────────────
    if joints[12].any() and joints[16].any():
        joints[0]  = (joints[12] + joints[16]) / 2   # spine base
    if joints[4].any() and joints[8].any():
        joints[20] = (joints[4]  + joints[8])  / 2   # spine shoulder
    if joints[0].any() and joints[20].any():
        joints[1]  = (joints[0]  + joints[20]) / 2   # spine mid
    if joints[6].any() and joints[7].any():
        joints[21] = joints[7]  + (joints[7]  - joints[6])  * 0.5  # left hand tip
    if joints[10].any() and joints[11].any():
        joints[23] = joints[11] + (joints[11] - joints[10]) * 0.5  # right hand tip
    # Neck: midpoint of head and spine shoulder
    if joints[3].any() and joints[20].any():
        joints[2]  = (joints[3]  + joints[20]) / 2

    # ── Center on spine base ─────────────────────────────────────────────
    spine_base = joints[0].copy()
    joints -= spine_base

    # ── Normalize to body scale ──────────────────────────────────────────
    # Use the skeleton height (spine base to head) as the scale reference.
    # This mimics NTU's metric-space normalization without needing depth.
    scale = np.linalg.norm(joints[3])  # distance head → spine base (after centering)
    if scale > 1e-3:
        joints /= scale

    return joints


def compute_bone(joints: np.ndarray) -> np.ndarray:
    """joints: (T, 25, 3) → bones: (T, 25, 3)"""
    bones = np.zeros_like(joints)
    for child, parent in NTU_BONE_PAIRS:
        bones[:, child] = joints[:, child] - joints[:, parent]
    return bones


def build_input_tensor(window: np.ndarray) -> torch.Tensor:
    """
    window : (T, 25, 3)  raw joint positions
    Returns : (1, 3, T, 25, 2)  ready for LG-SGNet forward()
                 N  C  T  V   M
    Model trained with num_person=2; second person padded with zeros.
    """
    # (T, V, C) → (C, T, V) → (1, C, T, V, 2)
    x = window.transpose(2, 0, 1)               # (3, T, 25)
    x = x[np.newaxis, :, :, :, np.newaxis]      # (1, 3, T, 25, 1)
    # Pad second person with zeros; model trained with num_person=2
    zeros = np.zeros_like(x)
    x = np.concatenate([x, zeros], axis=-1)     # (1, 3, T, 25, 2)
    return torch.from_numpy(x)


def compute_motion(sequence: np.ndarray) -> np.ndarray:
    """sequence: (T, 25, 3) → motion: (T, 25, 3). Frame 0 motion = 0."""
    motion = np.zeros_like(sequence)
    motion[1:] = sequence[1:] - sequence[:-1]
    return motion


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(models: dict, window: np.ndarray, device: torch.device) -> dict:
    """
    Run 4-modality inference on a single window of skeleton frames.

    Args:
        models : dict with keys 'joint', 'bone', 'joint_motion', 'bone_motion'
        window : np.ndarray of shape (T, 25, 3)
        device : torch device

    Returns dict with:
        'predicted_class'  : int
        'is_fall'          : bool
        'fall_probability' : float (after softmax)
        'scores_raw'       : np.ndarray (60,)
    """
    joints       = window.copy()
    bones        = compute_bone(joints)
    joint_motion = compute_motion(joints)
    bone_motion  = compute_motion(bones)

    modalities = {
        'joint':        joints,
        'bone':         bones,
        'joint_motion': joint_motion,
        'bone_motion':  bone_motion,
    }

    raw_scores = {}
    for name, data in modalities.items():
        tensor = build_input_tensor(data).to(device)
        out = models[name](tensor)             # (1, num_class)
        raw_scores[name] = out.cpu().numpy()[0] # (num_class,)

    # Weighted ensemble
    ensemble = (
        raw_scores['joint']        * ALPHA[0] +
        raw_scores['bone']         * ALPHA[1] +
        raw_scores['joint_motion'] * ALPHA[2] +
        raw_scores['bone_motion']  * ALPHA[3]
    )

    probs           = softmax(ensemble)
    predicted_class = int(np.argmax(ensemble))

    # Trigger on staggering (41) OR fall down (42)
    is_fall             = (predicted_class in FALL_CLASSES)
    fall_score_combined = max(ensemble[c] for c in FALL_CLASSES)
    fall_prob           = float(sum(probs[c] for c in FALL_CLASSES))

    sorted_idx  = np.argsort(ensemble)[::-1]
    fall_rank   = min(
        int(np.where(sorted_idx == c)[0][0]) + 1
        for c in FALL_CLASSES
    )

    top_score     = ensemble[sorted_idx[0]]
    score_range   = max(top_score - ensemble.min(), 1e-6)
    fall_relative = float(np.clip(
        (fall_score_combined - ensemble.min()) / score_range, 0, 1))

    return {
        'predicted_class':  predicted_class,
        'is_fall':          is_fall,
        'fall_probability': fall_prob,
        'fall_rank':        fall_rank,
        'fall_relative':    fall_relative,
        'fall_score_raw':   float(fall_score_combined),
        'top_score_raw':    float(top_score),
        'scores_raw':       ensemble,
    }


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, result: dict, fps: float, skeleton_detected: bool):
    h, w = frame.shape[:2]

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Skeleton status
    skel_color = (0, 255, 0) if skeleton_detected else (0, 100, 255)
    skel_text  = "Skeleton OK" if skeleton_detected else "No skeleton"
    cv2.putText(frame, skel_text, (w // 2 - 70, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, skel_color, 2)

    if result is None:
        cv2.putText(frame, "Collecting frames...", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return frame

    fall_relative = result['fall_relative']   # 0.0 – 1.0
    fall_rank     = result['fall_rank']        # 1 = top prediction
    is_fall       = result['is_fall']

    # Fall alert box
    if is_fall:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "FALL DETECTED!", (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    # Fall score bar (relative to top scorer)
    bar_x, bar_y, bar_w, bar_h = 10, h - 75, 300, 20
    bar_fill  = int(bar_w * fall_relative)
    bar_color = (0, 0, 255) if fall_relative > 0.8 else (0, 200, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)
    cv2.putText(frame, f"Fall score: {fall_relative:.2%}",
                (bar_x + bar_w + 10, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Fall rank bar (rank 1 = predicted fall, rank 60 = least likely)
    bar_y2    = h - 45
    rank_fill = int(bar_w * (1 - (fall_rank - 1) / 59))  # invert: rank1 = full bar
    rank_color = (0, 0, 255) if fall_rank == 1 else (
                 (0, 165, 255) if fall_rank <= 3 else (0, 200, 100))
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + rank_fill, bar_y2 + bar_h), rank_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h), (150, 150, 150), 1)
    cv2.putText(frame, f"Fall rank: #{fall_rank}/60",
                (bar_x + bar_w + 10, bar_y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-time fall detection with LG-SGNet")
    parser.add_argument('--joint-model',        required=True, help='Path to joint model .pt')
    parser.add_argument('--bone-model',         required=True, help='Path to bone model .pt')
    parser.add_argument('--joint-motion-model', required=True, help='Path to joint motion model .pt')
    parser.add_argument('--bone-motion-model',  required=True, help='Path to bone motion model .pt')
    parser.add_argument('--source',  default=0,  help='Camera index, RTSP URL, or video file path')
    parser.add_argument('--window-size', default=64, type=int,
                        help='Sliding window size in frames (default: 32, NTU trained on 64)')
    parser.add_argument('--resize', default=320, type=int,
                        help='Resize frame width for YOLO (default: 320, lower = faster)')
    parser.add_argument('--smooth-alpha', default=0.4, type=float,
                        help='EMA smoothing factor for keypoints 0.1-1.0 (default: 0.4, lower=smoother)')
    parser.add_argument('--num-class', default=60, type=int, help='Number of classes model was trained on')
    parser.add_argument('--window',  default=64,  type=int, help='Sliding window size (frames)')
    parser.add_argument('--stride',  default=8,   type=int, help='Inference stride (run every N new frames)')
    parser.add_argument('--device',  default='cpu',       help='Device for LG-SGNet models: cuda / cpu (default: cpu)')
    parser.add_argument('--yolo-device', default='0',     help='Device for YOLO: 0 (GPU) or cpu (default: 0)')
    parser.add_argument('--save-video', default='',       help='Path to save output video (optional)')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt',
                        help='Path to YOLO pose model (default: yolo11n-pose.pt)')
    args = parser.parse_args()

    global WINDOW_SIZE
    WINDOW_SIZE = args.window_size
    print(f"[INFO] Window size: {WINDOW_SIZE} frames")

    # Device for LG-SGNet
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA not available for PyTorch, falling back to CPU for LG-SGNet.")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"[INFO] LG-SGNet device: {device}")

    # Device for YOLO
    yolo_device = args.yolo_device
    if yolo_device == '0' and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU for YOLO.")
        yolo_device = 'cpu'
    print(f"[INFO] YOLO device: {yolo_device}")

    # Load models
    print("[INFO] Loading models...")
    models = {
        'joint':        load_model(args.joint_model,        args.num_class, device),
        'bone':         load_model(args.bone_model,         args.num_class, device),
        'joint_motion': load_model(args.joint_motion_model, args.num_class, device),
        'bone_motion':  load_model(args.bone_motion_model,  args.num_class, device),
    }
    print("[INFO] All models loaded.")

    # YOLO11-pose model
    print(f"[INFO] Loading pose model: {args.pose_model}")
    pose_model = YOLO(args.pose_model)


    # Video capture
    if str(args.source).isdigit():
        source = int(args.source)
        cap    = cv2.VideoCapture(source)
    else:
        source = args.source
        # Try FFmpeg backend first (handles MJPEG/HTTP streams like DroidCam)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        print("For DroidCam, try these URLs:")
        print(f"  http://<ip>:<port>/mjpegfeed")
        print(f"  http://<ip>:<port>/video")
        print(f"  http://<ip>:<port>/mjpegfeed?res=640x480")
        sys.exit(1)

    # For network streams, set buffer size to 1 to always get the latest frame
    if not str(args.source).isdigit():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {args.source}  |  {fw}x{fh} @ {fps_src:.1f} fps")

    # Optional video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_src, (fw, fh))
        print(f"[INFO] Saving output to: {args.save_video}")

    # ── Shared state between main thread and inference thread ────────────
    # ── COCO skeleton connections (for drawing) ───────────────────────────
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),   # nose → eyes → ears
        (5, 6),                             # shoulders
        (5, 7), (7, 9),                     # left arm
        (6, 8), (8, 10),                    # right arm
        (5, 11), (6, 12), (11, 12),         # torso
        (11, 13), (13, 15),                 # left leg
        (12, 14), (14, 16),                 # right leg
    ]

    # ── Shared state ───────────────────────────────────────────────────────
    # Between main thread ↔ YOLO thread
    latest_frame      = None          # most recent raw frame for YOLO to process
    frame_lock        = threading.Lock()
    pose_result       = None          # latest keypoints + drawn frame from YOLO
    pose_result_lock  = threading.Lock()
    pose_ready        = threading.Event()

    # Between YOLO thread ↔ inference thread
    skeleton_buffer   = deque(maxlen=WINDOW_SIZE)
    buffer_lock       = threading.Lock()
    frames_since_inf  = [0]           # mutable counter shared across threads
    inference_ready   = threading.Event()

    # Between inference thread ↔ main thread
    last_result       = None
    result_lock       = threading.Lock()

    stop_event        = threading.Event()

    # ── YOLO pose thread ───────────────────────────────────────────────────
    def yolo_worker():
        # Keypoint smoothing — exponential moving average over last N YOLO frames
        SMOOTH_ALPHA      = args.smooth_alpha  # lower = smoother but more lag (0.1–1.0)
        smoothed_kpts     = None          # (17, 3) smoothed keypoints in small-frame coords
        smoothed_kpts_orig = None         # (17, 3) smoothed keypoints in original-frame coords
        nonlocal pose_result
        while not stop_event.is_set():
            # Wait for a new frame from main thread
            triggered = pose_ready.wait(timeout=0.5)
            if not triggered:
                continue
            pose_ready.clear()

            with frame_lock:
                if latest_frame is None:
                    continue
                frame_copy = latest_frame.copy()

            # Resize for faster YOLO inference (keypoints scaled back after)
            orig_h, orig_w = frame_copy.shape[:2]
            scale_w = args.resize / orig_w
            small   = cv2.resize(frame_copy, (args.resize, int(orig_h * scale_w)))

            # Run YOLO pose estimation on smaller frame
            yolo_results = pose_model(small, verbose=False, device=yolo_device)

            detected   = False
            keypoints  = None
            drawn      = frame_copy.copy()

            if yolo_results and yolo_results[0].keypoints is not None:
                kpts = yolo_results[0].keypoints.data
                if kpts.shape[0] > 0:
                    detected  = True
                    raw_kpts  = kpts[0].cpu().numpy()  # (17, 3) in small frame coords

                    # ── Exponential moving average smoothing ─────────────
                    # Smooths out per-frame jitter while preserving real motion.
                    # Only smooth x,y — keep confidence (col 2) as-is.
                    if smoothed_kpts is None:
                        smoothed_kpts = raw_kpts.copy()
                    else:
                        # Only smooth joints with sufficient confidence
                        high_conf = raw_kpts[:, 2] > 0.3
                        smoothed_kpts[high_conf, :2] = (
                            SMOOTH_ALPHA * raw_kpts[high_conf, :2] +
                            (1 - SMOOTH_ALPHA) * smoothed_kpts[high_conf, :2]
                        )
                        smoothed_kpts[:, 2] = raw_kpts[:, 2]  # always use latest conf

                    keypoints = smoothed_kpts

                    # Scale smoothed keypoints back to original frame resolution
                    keypoints_orig = keypoints.copy()
                    keypoints_orig[:, 0] /= scale_w   # x
                    keypoints_orig[:, 1] /= scale_w   # y

                    joints = coco_to_ntu_skeleton(keypoints)  # use small coords (normalized anyway)
                    with buffer_lock:
                        skeleton_buffer.append(joints)
                        buf_len = len(skeleton_buffer)

                    # Signal inference thread every stride frames
                    frames_since_inf[0] += 1
                    if frames_since_inf[0] >= args.stride and buf_len == WINDOW_SIZE:
                        inference_ready.set()
                        frames_since_inf[0] = 0

                    # Draw skeleton onto drawn frame using original-res coords
                    for coco_idx in range(17):
                        x_px, y_px, conf = keypoints_orig[coco_idx]
                        if conf > 0.3:
                            cv2.circle(drawn, (int(x_px), int(y_px)), 4, (0, 255, 0), -1)
                    for a, b in POSE_CONNECTIONS:
                        if a < 17 and b < 17:
                            xa, ya, ca = keypoints_orig[a]
                            xb, yb, cb = keypoints_orig[b]
                            if ca > 0.3 and cb > 0.3:
                                cv2.line(drawn,
                                         (int(xa), int(ya)),
                                         (int(xb), int(yb)),
                                         (0, 180, 255), 2)

            # Reset smoother when person disappears to avoid stale keypoints
            if not detected:
                smoothed_kpts = None

            with pose_result_lock:
                pose_result = {'detected': detected, 'frame': drawn}

    # ── LG-SGNet inference thread ──────────────────────────────────────────
    def inference_worker():
        nonlocal last_result
        while not stop_event.is_set():
            triggered = inference_ready.wait(timeout=1.0)
            if not triggered:
                continue
            inference_ready.clear()

            with buffer_lock:
                if len(skeleton_buffer) < WINDOW_SIZE:
                    continue
                window = np.stack(list(skeleton_buffer), axis=0)

            result = run_inference(models, window, device)

            with result_lock:
                last_result = result

            # ── Debug ────────────────────────────────────────────────────
            scores    = result['scores_raw']
            top5_idx  = np.argsort(scores)[-5:][::-1]
            top5_str  = "  ".join(f"cls{i}={scores[i]:.3f}" for i in top5_idx)
            disp      = np.linalg.norm(window[-1] - window[0], axis=-1).mean()
            vel       = np.linalg.norm(np.diff(window, axis=0), axis=-1).mean()
            pred_name = NTU60_CLASSES.get(result['predicted_class'], '?')
            print(f"[DEBUG] top5: {top5_str} | cls41={scores[41]:.3f} cls42={scores[42]:.3f} | "
                  f"rank={result['fall_rank']} | pred={result['predicted_class']}({pred_name})")
            print(f"[DEBUG] disp={disp:.4f} vel={vel:.4f} | "
                  f"joint: min={window.min():.3f} max={window.max():.3f} std={window.std():.3f}")

            if result['is_fall']:
                trigger = NTU60_CLASSES.get(result['predicted_class'], '?')
                print(f"[ALERT] FALL DETECTED via '{trigger}' (cls{result['predicted_class']})  "
                      f"fall_prob={result['fall_probability']:.3f}")

    # Start background threads
    yolo_thread      = threading.Thread(target=yolo_worker,      daemon=True)
    inference_thread = threading.Thread(target=inference_worker, daemon=True)
    yolo_thread.start()
    inference_thread.start()

    # ── Main loop — only does capture + display, never blocks ─────────────
    prev_time = time.time()
    print("[INFO] Starting real-time inference. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            # Hand frame to YOLO thread (drop if YOLO is still busy)
            if not pose_ready.is_set():
                with frame_lock:
                    latest_frame = frame
                pose_ready.set()

            # Get latest pose-drawn frame (non-blocking)
            with pose_result_lock:
                pr = pose_result

            if pr is not None:
                display_frame    = pr['frame'].copy()
                skeleton_detected = pr['detected']
            else:
                display_frame    = frame.copy()
                skeleton_detected = False

            # FPS
            now     = time.time()
            fps_cur = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Read latest inference result
            with result_lock:
                current_result = last_result

            display_frame = draw_overlay(
                display_frame, current_result, fps_cur, skeleton_detected)

            if writer:
                writer.write(display_frame)

            cv2.imshow("LG-SGNet Fall Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        pose_ready.set()       # unblock yolo_worker if waiting
        inference_ready.set()  # unblock inference_worker if waiting
        yolo_thread.join(timeout=3)
        inference_thread.join(timeout=3)
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == '__main__':
    main()