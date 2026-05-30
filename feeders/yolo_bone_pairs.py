# YOLO 17-joint bone pairs (0-based indices, matches graph/yolo.py inward edges)
# Each tuple (v1, v2): bone vector = joint[v1] - joint[v2]
#
# Joint index reference:
#  0 Nose        1 L.Eye      2 R.Eye      3 L.Ear      4 R.Ear
#  5 L.Shoulder  6 R.Shoulder 7 L.Elbow    8 R.Elbow    9 L.Wrist
# 10 R.Wrist    11 L.Hip     12 R.Hip     13 L.Knee    14 R.Knee
# 15 L.Ankle    16 R.Ankle

yolo_pairs = (
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
)
