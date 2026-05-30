import numpy as np

data = np.load('data/ntu/NTU60_CS.npz')
labels = np.where(data['y_train'] > 0)[1]

unique, counts = np.unique(labels, return_counts=True)
for cls, cnt in zip(unique, counts):
    marker = " ← FALL" if cls == 42 else ""
    print(f"Class {cls:2d}: {cnt} samples{marker}")

print(f"\nFall samples: {counts[42]}")
print(f"Mean per class: {counts.mean():.0f}")
print(f"Imbalance ratio: {counts.mean()/counts[42]:.1f}x")