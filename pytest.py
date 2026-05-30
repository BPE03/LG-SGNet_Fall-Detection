import numpy as np

sample = np.load('data/nturgbd_raw_yolo/fall/S001C001P001R002A043_rgb.npy')
print(f"shape: {sample.shape}")          # (num_frames, 17, 3)
print(f"min={sample.min():.4f}  max={sample.max():.4f}  std={sample.std():.4f}")
print(f"first frame:\n{sample[0]}")      # (17, 3) — x, y, conf