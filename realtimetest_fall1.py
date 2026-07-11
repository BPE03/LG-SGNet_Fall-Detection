"""
Real-time Fall Detection using LG-SGNet (joint modality only)
==============================================================
Requirements:
    pip install torch opencv-python ultralytics numpy

Usage:
    python realtime_fall_detection.py \
        --joint-model      work_dir/ntu60/fall/joint/best_model.pt \
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

# YOLO11-pose outputs 17 COCO keypoints (0-indexed)
NUM_JOINTS  = 17
NUM_COORDS  = 3   # x, y, z (MediaPipe gives depth z estimate)
WINDOW_SIZE = 64  # frames expected by LG-SGNet

BINARY_CLASSES = {
    0: "non-fall",
    1: "fall",
}

def ts() -> str:
    return time.strftime('%H:%M:%S') + f'.{int(time.time() * 1000) % 1000:03d}'

# ── Helper: import Model from LG-SGNet repo ────────────────────────────────────
def load_model(checkpoint_path: str, num_class: int = 2, device: torch.device = None):
    """
    Dynamically imports Model from the local LG-SGNet repo and loads weights.
    Assumes this script is run from the root of the LG-SGNet directory.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # LG-SGNet model is in model/LG_SGNet.py (or similar)
        import importlib
        model_module = importlib.import_module('model.LG-SGNet-17')
        model = model_module.Model(num_class=num_class, num_point=17, num_person=1,
                                   graph='graph.yolo.Graph',
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
def normalize_skeleton_sequence(buffer_np):
    sk = buffer_np.copy()
    xy = sk[:, :, :2]

    # Translate: center on hip midpoint per frame
    hc  = (xy[:, 11] + xy[:, 12]) / 2.0    # (T, 2)
    xy -= hc[:, np.newaxis, :]

    # Scale: mean shoulder width across whole sequence
    d   = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)   # (T,)
    sc  = d[d > 1e-5].mean() if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        xy /= sc

    sk[:, :, :2] = xy
    return sk


def build_input_tensor(window: np.ndarray) -> torch.Tensor:
    """
    window : (T, 17, 3)  raw joint positions
    Returns : (1, 3, T, 17, 1)  ready for LG-SGNet forward()
                 N  C  T  V   M
    Model trained with num_person=1.
    """
    # (T, V, C) → (C, T, V) → (1, C, T, V, 1)
    x = window.transpose(2, 0, 1)               # (3, T, 17)
    x = x[np.newaxis, :, :, :, np.newaxis]      # (1, 3, T, 17, 1)
    return torch.from_numpy(x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, window: np.ndarray, device: torch.device) -> dict:
    """
    Run joint modality inference on a single window of skeleton frames.

    Args:
        model : LG-SGNet model
        window : np.ndarray of shape (T, 17, 3)
        device : torch device

    Returns dict with:
        'predicted_class'  : int
        'is_fall'          : bool
        'fall_probability' : float (after softmax)
        'scores_raw'       : np.ndarray (2,)
    """
    joints = window.copy()
    joints = normalize_skeleton_sequence(joints)  # in-place normalization

    # ── Sanity check: joint values should be roughly in [-3, 3] ──────
    # If values are way out of range, normalization failed (e.g. YOLO lost
    # the person mid-window). Return None so the caller can flush the buffer
    # rather than running inference on corrupted data.
    jmin, jmax = joints.min(), joints.max()
    # if jmax > 10.0 or jmin < -10.0:
    #     print(f"[WARN] Skipping inference — joint values out of NTU range: "
    #           f"min={jmin:.2f} max={jmax:.2f}. Flushing buffer.")
    #     return None

    tensor = build_input_tensor(joints).to(device)
    out = model(tensor)             # (1, num_class)
    raw_scores = out.cpu().numpy()[0] # (num_class,)
    # Debug: helps identify model behavior
    print(f"[{ts()}] [DEBUG] joint: cls0={raw_scores[0]:.3f}  cls1={raw_scores[1]:.3f}")

    probs           = softmax(raw_scores)
    predicted_class = int(np.argmax(raw_scores))

    # Trigger on fall (1)
    is_fall             = (predicted_class == 1)
    fall_score_combined = raw_scores[1] if len(raw_scores) > 1 else 0
    fall_prob           = float(probs[1]) if len(probs) > 1 else 0

    sorted_idx  = np.argsort(raw_scores)[::-1]
    fall_rank   = int(np.where(sorted_idx == 1)[0][0]) + 1  # actual rank of class 1

    top_score     = raw_scores[sorted_idx[0]]
    score_range   = max(top_score - raw_scores.min(), 1e-6)
    fall_relative = float(np.clip(
        (fall_score_combined - raw_scores.min()) / score_range, 0, 1))

    return {
        'predicted_class':  predicted_class,
        'is_fall':          is_fall,
        'fall_probability': fall_prob,
        'fall_rank':        fall_rank,
        'fall_relative':    fall_relative,
        'fall_score_raw':   float(fall_score_combined),
        'top_score_raw':    float(top_score),
        'scores_raw':       raw_scores,
    }


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, result: dict, fps: float, skeleton_detected: bool, fall_threshold: float):
    h, w = frame.shape[:2]

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Resolution
    cv2.putText(frame, f"Resolution: {w}x{h}", (w - 240, 35),
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

    fall_relative = result['fall_probability']   # already softmax prob 0.0–1.0
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
    bar_color = (0, 0, 255) if fall_relative > fall_threshold else (0, 200, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)
    cv2.putText(frame, f"Fall prob: {fall_relative:.2%}",
                (bar_x + bar_w + 10, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Fall rank bar (rank 1 = predicted fall, rank 2 = non-fall)
    bar_y2    = h - 45
    rank_fill = int(bar_w * (1 - (fall_rank - 1) / 1))  # invert: rank1 = full bar
    rank_color = (0, 0, 255) if fall_rank == 1 else (0, 200, 100)
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + rank_fill, bar_y2 + bar_h), rank_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h), (150, 150, 150), 1)
    cv2.putText(frame, f"Fall rank: #{fall_rank}/2",
                (bar_x + bar_w + 10, bar_y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-time fall detection with LG-SGNet")
    parser.add_argument('--joint-model',        required=True, help='Path to joint model .pt')
    parser.add_argument('--source',  default=0,  help='Camera index, RTSP URL, or video file path')
    parser.add_argument('--window-size', default=64, type=int,
                        help='Sliding window size in frames (default: 32, NTU trained on 64)')
    parser.add_argument('--resize', default=320, type=int,
                        help='Resize frame width for YOLO (default: 320, lower = faster)')
    parser.add_argument('--smooth-alpha', default=0.4, type=float,
                        help='EMA smoothing factor for keypoints 0.1-1.0 (default: 0.4, lower=smoother)')
    parser.add_argument('--num-class', default=2, type=int, help='Number of classes model was trained on')
    parser.add_argument('--window',  default=64,  type=int, help='Sliding window size (frames)')
    parser.add_argument('--stride',  default=8,   type=int, help='Inference stride (run every N new frames)')
    parser.add_argument('--vote-buffer', default=5, type=int,
                        help='Number of recent predictions to vote over (default: 5)')
    parser.add_argument('--fall-threshold', default=0.6, type=float,
                        help='Fraction of vote buffer that must predict fall to trigger alert (default: 0.6)')
    parser.add_argument('--device',  default='cpu',       help='Device for LG-SGNet models: cuda / cpu (default: cpu)')
    parser.add_argument('--yolo-device', default='0',     help='Device for YOLO: 0 (GPU) or cpu (default: 0)')
    parser.add_argument('--save-video', default='',       help='Path to save output video (optional)')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt',
                        help='Path to YOLO pose model (default: yolo11n-pose.pt)')
    parser.add_argument('--display-width', default=0, type=int,
                        help='Max width for displayed window (0 = original, maintains aspect ratio)')
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
    print("[INFO] Loading model...")
    model = load_model(args.joint_model, args.num_class, device)
    print("[INFO] Model loaded.")

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
    missing_frames    = [0]           # consecutive frames with no detection
    MISSING_GRACE     = 10           # frames of no-detection before clearing buffer
    inference_ready   = threading.Event()

    # Between inference thread ↔ main thread
    last_result       = None
    result_lock       = threading.Lock()

    # Temporal voting buffer — smooths out single-window false positives
    vote_buffer       = deque(maxlen=args.vote_buffer)
    vote_lock         = threading.Lock()

    stop_event        = threading.Event()

    # ── YOLO pose thread ───────────────────────────────────────────────────
    def yolo_worker():
        # Keypoint smoothing — exponential moving average over last N YOLO frames
        SMOOTH_ALPHA      = args.smooth_alpha  # lower = smoother but more lag (0.1–1.0)
        smoothed_kpts     = None          # (17, 3) smoothed keypoints in small-frame coords
        nonlocal pose_result
        nonlocal vote_buffer, vote_lock
        nonlocal missing_frames
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

                    kf = np.zeros((NUM_JOINTS, 3), np.float32)
                    kf[:, :2] = keypoints_orig[:, :2]
                    kf[:, 2]  = raw_kpts[:, 2]   # confidence
                    with buffer_lock:
                        skeleton_buffer.append(kf)
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

            # Reset smoother AND skeleton buffer when person disappears.
            # Stale pre-disappearance frames in the buffer would mix with fresh
            # post-reappearance frames, corrupting the 64-frame window.
            # Only clear buffer after MISSING_GRACE consecutive missing frames.
            # Brief losses (person on ground, partial occlusion) are bridged by
            # repeating the last known skeleton so inference keeps running.
            if not detected:
                missing_frames[0] += 1
                if missing_frames[0] <= MISSING_GRACE:
                    # Bridge gap: repeat last known frame if buffer has data
                    with buffer_lock:
                        if len(skeleton_buffer) > 0:
                            skeleton_buffer.append(skeleton_buffer[-1])
                            frames_since_inf[0] += 1
                            if frames_since_inf[0] >= args.stride and len(skeleton_buffer) == WINDOW_SIZE:
                                inference_ready.set()
                                frames_since_inf[0] = 0
                else:
                    # Person truly gone — reset everything
                    smoothed_kpts = None
                    with buffer_lock:
                        skeleton_buffer.clear()
                    with vote_lock:
                        vote_buffer.clear()
                    missing_frames[0] = 0
            else:
                missing_frames[0] = 0

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

            result = run_inference(model, window, device)

            # Normalization failed — poisoned frames in buffer, flush it so
            # we start fresh rather than keep inferring on bad data.
            if result is None:
                with buffer_lock:
                    skeleton_buffer.clear()
                continue

            with result_lock:
                last_result = result

            # ── Temporal voting ──────────────────────────────────────────
            # Accumulate fall probability over recent windows and only
            # trigger if the mean exceeds fall_threshold. This prevents
            # single-window false positives from firing an alert.
            with vote_lock:
                vote_buffer.append(result['fall_probability'])
                vote_mean = float(np.mean(vote_buffer))
                voted_fall = (len(vote_buffer) == args.vote_buffer and
                              vote_mean >= args.fall_threshold)

            # Fast-clear on confident non-fall
            if result['fall_probability'] < 0.2:  # tune threshold
                with vote_lock:
                    vote_buffer.clear()

            # Override is_fall in the result with the voted decision
            with result_lock:
                last_result = dict(last_result)   # shallow copy
                last_result['is_fall']          = voted_fall
                last_result['fall_probability'] = vote_mean

            # ── Debug ────────────────────────────────────────────────────
            scores    = result['scores_raw']
            top2_idx  = np.argsort(scores)[-2:][::-1]
            top2_str  = "  ".join(f"cls{i}={scores[i]:.3f}" for i in top2_idx)
            disp      = np.linalg.norm(window[-1] - window[0], axis=-1).mean()
            vel       = np.linalg.norm(np.diff(window, axis=0), axis=-1).mean()
            pred_name = BINARY_CLASSES.get(result['predicted_class'], '?')
            print(f"[DEBUG] top2: {top2_str} | "
                  f"rank={result['fall_rank']} | pred={result['predicted_class']}({pred_name})")
            print(f"[DEBUG] disp={disp:.4f} vel={vel:.4f} | "
                  f"joint: min={window.min():.3f} max={window.max():.3f} std={window.std():.3f}")
            print(f"[DEBUG] "
                f"buf={len(skeleton_buffer):2d}frames | "
                f"raw=[{result['scores_raw'][0]:.3f}, {result['scores_raw'][1]:.3f}] | "
                f"fall_prob={result['fall_probability']:.3f} | "
                f"vote_buf={list(round(x,3) for x in vote_buffer)} | "
                f"vote_mean={vote_mean:.3f} | "
                f"voted={'FALL' if voted_fall else 'ok'}")

            if result['is_fall']:
                trigger = BINARY_CLASSES.get(result['predicted_class'], '?')
                print(f"[ALERT] FALL DETECTED (cls{result['predicted_class']})  "
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
                display_frame, current_result, fps_cur, skeleton_detected,
                fall_threshold=args.fall_threshold)

            if writer:
                writer.write(display_frame)

            # Resize for display if --display-width is specified
            if args.display_width > 0 and display_frame.shape[1] > args.display_width:
                scale = args.display_width / display_frame.shape[1]
                new_h = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (args.display_width, new_h))

            cv2.imshow("LG-SGNet Fall Detection", display_frame)
            if str(args.source).endswith('.mp4'):
                wait_ms = max(1, int(1000 / fps_src))
            else:
                wait_ms = 1
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
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