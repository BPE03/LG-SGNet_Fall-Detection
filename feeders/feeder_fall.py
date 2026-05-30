import numpy as np

from torch.utils.data import Dataset

from feeders import tools


# ── Fall Detection Label Mapping (NTU RGB+D 60, 1-indexed) ───────────────────
# Fall     (binary label 1): 43 = fall down
# Non-fall (binary label 0): 8  = sit down
#                             9  = stand up
#                             27 = jump up
#                             42 = walking toward each other
FALL_LABELS     = {43}            # 1-indexed NTU action labels → binary 1
NON_FALL_LABELS = {8, 9, 27, 42}  # 1-indexed NTU action labels → binary 0
SELECTED_LABELS = FALL_LABELS | NON_FALL_LABELS  # all labels to keep

# # ── Fall Detection Label Mapping (NTU RGB+D 60, 1-indexed) ───────────────────
# # Fall     (binary label 1): 43 = fall down
# # Non-fall (binary label 0): all other NTU60 actions except 43
# FALL_LABELS     = {43}                        # 1-indexed NTU action labels → binary 1
# NON_FALL_LABELS = set(range(1, 61)) - FALL_LABELS  # 1-indexed NTU action labels → binary 0
# SELECTED_LABELS = set(range(1, 61))          # keep all NTU60 labels


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        Fall-detection feeder for LG-SGNet on NTU RGB+D 60.
        Filters samples to SELECTED_LABELS only and remaps labels to binary:
            1 = fall (NTU label 43)
            0 = non-fall (NTU labels 8, 9, 27, 42)

        :param data_path: path to NTU60_CS.npz or NTU60_CV.npz
        :param label_path: unused (kept for API compatibility)
        :param split: 'train' or 'test'
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)

        if self.split == 'train':
            raw_data  = npz_data['x_train']
            # y_train is one-hot → convert to 0-indexed integer labels first
            raw_label_0idx = np.where(npz_data['y_train'] > 0)[1]  # 0-indexed
            split_tag = 'train'
        elif self.split == 'test':
            raw_data  = npz_data['x_test']
            raw_label_0idx = np.where(npz_data['y_test'] > 0)[1]   # 0-indexed
            split_tag = 'test'
        else:
            raise NotImplementedError('data split only supports train/test')

        # Convert to 1-indexed NTU labels to match SELECTED_LABELS definition
        raw_label_1idx = raw_label_0idx + 1  # now 1-indexed (1–60)

        # ── Filter: keep only samples belonging to SELECTED_LABELS ────────────
        mask = np.isin(raw_label_1idx, list(SELECTED_LABELS))
        filtered_data      = raw_data[mask]
        filtered_label_1idx = raw_label_1idx[mask]

        # ── Remap to binary labels: fall=1, non-fall=0 ────────────────────────
        binary_label = np.where(np.isin(filtered_label_1idx, list(FALL_LABELS)), 1, 0)

        self.label       = binary_label
        self.sample_name = [f'{split_tag}_{i}' for i in range(len(filtered_data))]

        # Print class distribution for transparency
        n_fall     = int((binary_label == 1).sum())
        n_non_fall = int((binary_label == 0).sum())
        print(f'[FallFeeder:{split_tag}] Total={len(binary_label)}  '
              f'Fall(1)={n_fall}  Non-fall(0)={n_non_fall}')

        if self.debug:
            debug_size = 100
            filtered_data    = filtered_data[:debug_size]
            self.label       = self.label[:debug_size]
            self.sample_name = self.sample_name[:debug_size]
            print(f'[FallFeeder:{split_tag}] Debug mode: using first {debug_size} samples')

        N, T, _ = filtered_data.shape
        self.data = filtered_data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        #data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod