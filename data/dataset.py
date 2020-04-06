import torch.utils.data as data
import torch
import numpy as np
import os
import scipy.misc as misc


class DatasetFromFolder(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromFolder, self).__init__()
        self.LR_paths = get_image_path(file_path['LR'])
        self.x2_paths = get_image_path(file_path['HR'])

        assert(len(self.LR_paths) == len(self.x2_paths))

    def __getitem__(self, item):
        # lr = read_img(self.LR_paths[item])
        # x2 = read_img(self.x2_paths[item])
        # x4 = read_img(self.x4_paths[item])
        lr = np.load(self.LR_paths[item])
        hr = np.load(self.x2_paths[item])
        lr = lr[np.newaxis, :]
        hr = hr[np.newaxis, :]

        return torch.from_numpy(lr).float(), torch.from_numpy(hr).float()

    def __len__(self):
        return len(self.LR_paths)


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
