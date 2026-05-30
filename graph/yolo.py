import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

# COCO 17-joint layout (YOLO11n-pose)
# Index → Joint
#  0  Nose
#  1  Left Eye
#  2  Right Eye
#  3  Left Ear
#  4  Right Ear
#  5  Left Shoulder
#  6  Right Shoulder
#  7  Left Elbow
#  8  Right Elbow
#  9  Left Wrist
# 10  Right Wrist
# 11  Left Hip
# 12  Right Hip
# 13  Left Knee
# 14  Right Knee
# 15  Left Ankle
# 16  Right Ankle

num_node = 17
self_link = [(i, i) for i in range(num_node)]

# Inward edges: from peripheral joints toward the body centre.
# Convention matches ntu_rgb_d.py — edges point *toward* the root.
# Root chosen as mid-torso, approximated by left hip (11) / right hip (12).
# Tree structure (child → parent):
#   Face
#     1 (L.Eye)  → 0 (Nose)
#     2 (R.Eye)  → 0 (Nose)
#     3 (L.Ear)  → 1 (L.Eye)
#     4 (R.Ear)  → 2 (R.Eye)
#   Upper body
#     0 (Nose)       → 6 (R.Shoulder)   # head anchored to right shoulder
#     5 (L.Shoulder) → 6 (R.Shoulder)   # shoulder girdle
#     6 (R.Shoulder) → 12 (R.Hip)       # torso right side
#     5 (L.Shoulder) → 11 (L.Hip)       # torso left side
#     7 (L.Elbow)    → 5 (L.Shoulder)
#     8 (R.Elbow)    → 6 (R.Shoulder)
#     9 (L.Wrist)    → 7 (L.Elbow)
#    10 (R.Wrist)    → 8 (R.Elbow)
#   Lower body
#    11 (L.Hip)  → 12 (R.Hip)           # hip girdle (root at R.Hip)
#    13 (L.Knee) → 11 (L.Hip)
#    14 (R.Knee) → 12 (R.Hip)
#    15 (L.Ankle)→ 13 (L.Knee)
#    16 (R.Ankle)→ 14 (R.Knee)
inward_ori_index = [
    # face
    (1, 0), (2, 0), (3, 1), (4, 2),
    # head to torso
    (0, 6),
    # shoulder girdle & torso
    (5, 6), (6, 12), (5, 11),
    # left arm
    (7, 5), (9, 7),
    # right arm
    (8, 6), (10, 8),
    # hip girdle
    (11, 12),
    # left leg
    (13, 11), (15, 13),
    # right leg
    (14, 12), (16, 14),
]

inward  = inward_ori_index          # already 0-based
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A