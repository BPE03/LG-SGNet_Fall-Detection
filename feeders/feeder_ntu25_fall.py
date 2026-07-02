import numpy as np
import random
from torch.utils.data import Dataset

from feeders import tools

# Pasangan joint kiri-kanan untuk flip augmentasi (COCO format)
FLIP_PAIRS = [
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (21, 23), (22, 24),
]

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_flip=False, random_speed=False, random_rot=False, random_noise=False,
                 window_size=64, normalization=False, debug=False, use_mmap=False, bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
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
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_flip  = random_flip
        self.random_speed = random_speed
        self.random_noise = random_noise
        self.random_rot   = random_rot
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
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)
        # print(npz_data['x_train'].shape)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        # DEBUG DATASET REDUCTION
        # print("Debug on")
        # debug_size = 1000
        # self.data = self.data[:debug_size]
        # self.label = self.label[:debug_size]
        # self.sample_name = self.sample_name[:debug_size]
            
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 1, 25, 3)).transpose(0, 4, 1, 3, 2)

        # Drop samples whose every frame is zero (corrupted source data)
        # valid_mask = self.data.reshape(N, -1).any(axis=1)
        # if not valid_mask.all():
        #     n_dropped = int((~valid_mask).sum())
        #     print(f'[Feeder] Dropping {n_dropped} all-zero sample(s) from {self.split} split.')
        #     self.data = self.data[valid_mask]
        #     self.label = self.label[valid_mask]
        #     self.sample_name = [s for s, v in zip(self.sample_name, valid_mask) if v]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def _count_valid_frames(self, x):
        """
        Hitung frame yang punya data (bukan semua-nol).
        NTU: frame kosong = semua koordinat nol karena padding.
        """
        spatial = x[:, :, :, 0]          # (C, T, V)
        valid = int((spatial != 0).any(axis=(0, 2)).sum())
        return max(valid, 1)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        # print(f"Original shape: {data_numpy.shape}")  # (C, T, V, M)
        label = self.label[index]
        #data_numpy = np.array(data_numpy)
        #valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        valid_frame_num = self._count_valid_frames(data_numpy)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.random_shift:
            data_numpy = self._shift(data_numpy)
        if self.random_move:
            data_numpy = self._rotate3d(data_numpy)
        if self.random_flip:
            data_numpy = self._flip(data_numpy)
        if self.random_speed:
            data_numpy = self._speed_perturb(data_numpy)
        if self.random_noise:
            data_numpy = self._add_noise(data_numpy)
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
    
    def _shift(self, x):
        """Random global translation (translasi dalam meter, misal ±0.1m)."""
        x = x.copy()
        x[0] += random.uniform(-0.1, 0.1)  # x
        x[1] += random.uniform(-0.1, 0.1)  # y
        x[2] += random.uniform(-0.1, 0.1)  # z
        return x

    def _rotate3d(self, x):
        """
        Random rotasi 3D di sekitar sumbu Y (vertikal, orang berdiri).
        Tambahan rotasi kecil pada sumbu X dan Z untuk variasi.
        """
        x   = x.copy()
        # Rotasi utama: sumbu Y (yaw) ±30 derajat
        yaw   = random.uniform(-0.52, 0.52)   # ±30°
        # Rotasi minor: pitch & roll ±10°
        pitch = random.uniform(-0.17, 0.17)
        roll  = random.uniform(-0.17, 0.17)

        # Matriks rotasi Ry (yaw, sumbu Y)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
        R  = Ry @ Rx @ Rz   # (3, 3)

        # x shape: (C=3, T, V=25, M=1)
        coords = x[:, :, :, 0]        # (3, T, 25)
        coords = np.einsum('ij,jkl->ikl', R, coords)  # (3, T, 25)
        x[:, :, :, 0] = coords
        return x

    def _flip(self, x):
        """Horizontal flip with COCO joint pair swap (50% probability)."""
        if random.random() > 0.5:
            return x
        x = x.copy()
        x[0] = -x[0]
        for l_idx, r_idx in FLIP_PAIRS:
            x[:, :, [l_idx, r_idx], :] = x[:, :, [r_idx, l_idx], :]
        return x

    def _speed_perturb(self, x):
        """Random temporal speed change (0.75×–1.25×) via resampling."""
        C, T, V, M = x.shape
        factor  = random.uniform(0.75, 1.25)
        new_len = max(1, int(T * factor))
        src_idx = np.linspace(0, T - 1, new_len, dtype=int)
        tgt_idx = np.linspace(0, new_len - 1, T, dtype=int)
        return x[:, src_idx, :, :][:, tgt_idx, :, :]
    
    def _add_noise(self, x):
        """Small Gaussian noise on x,y coordinates (sigma=0.01)."""
        x = x.copy()
        noise = np.random.normal(0, 0.01, x[:2].shape).astype(np.float32)
        x[:2] += noise
        return x

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