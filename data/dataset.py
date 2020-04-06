import torch.utils.data as data
import torch
import numpy as np
import os
import scipy.misc as misc


class DatasetFromFolder(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromFolder, self).__init__()
        self.Input_paths = get_image_path(file_path['Input'])
        self.S_paths = get_image_path(file_path['S'])
        self.T_paths = get_image_path(file_path['T'])
        self.GT_paths = get_image_path(file_path['GT'])

        assert(len(self.Input_paths) == len(self.S_paths) and
               len(self.S_paths) == len(self.T_paths) and
               len(self.T_paths) == len(self.GT_paths))

    def __getitem__(self, item):
        # lr = read_img(self.LR_paths[item])
        # x2 = read_img(self.x2_paths[item])
        # x4 = read_img(self.x4_paths[item])
        input = np.load(self.Input_paths[item])
        s = np.load(self.S_paths[item])
        t = np.load(self.T_paths[item])
        gt = np.load(self.GT_paths[item])
        input = input[np.newaxis, :]
        s = s[np.newaxis, :]
        t = t[np.newaxis, :]
        gt = gt[np.newaxis, :]

        return torch.from_numpy(input).float(), torch.from_numpy(s).float(), \
               torch.from_numpy(t).float(), torch.from_numpy(gt).float()

    def __len__(self):
        return len(self.input_paths)


def get_image_path(path):
    assert(os.path.isdir(path))
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            binary_path = os.path.join(dirpath, fname)
            files.append(binary_path)
    return files


def read_img(path):
    img = misc.imread(path)
    return img
