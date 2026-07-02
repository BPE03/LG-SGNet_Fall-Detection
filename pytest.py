import numpy as np

# # sample = np.load('data/nturgbd_raw_yolo/fall/S001C001P001R002A043_rgb.npy')
# sample = np.load('data/ntuYOLO1/balanced/joint/train_data.npy')[0]  # (T, 51)
# print(f"shape: {sample.shape}")          # (num_frames, 17, 3)
# print(f"min={sample.min():.4f}  max={sample.max():.4f}  std={sample.std():.4f}")
# print(f"first frame:\n{sample[0]}")      # (17, 3) — x, y, conf

import numpy as np
d = np.load('data/ntu25_data/balanced/joint/dataset.npz')
#d = np.load('data/ntu/fall_detection/balanced/joint/dataset.npz')
x = d['x_train']
y = d['y_train']
valid = np.array([(x[i] != 0).any(axis=1).sum() for i in range(len(x))])

print("=== Zero sequences ===")
print(f"count: {(valid == 0).sum()}")

print("\n=== Distribution of short sequences (<64) ===")
for threshold in [0, 10, 20, 30, 40, 50, 60, 64]:
    print(f"  < {threshold:3d} frames: {(valid < threshold).sum()}")

print("\n=== Class breakdown of short sequences ===")
labels = np.where(y > 0)[1]
short_mask = valid < 64
print(f"  fall (1)     : {(labels[short_mask] == 1).sum()}")
print(f"  not_fall (0) : {(labels[short_mask] == 0).sum()}")