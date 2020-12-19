from torch.utils import data
import os
import h5py
import torch
import numpy as np

class ToTensor:

    def __call__(self, sample):
        X, Y = sample
        return torch.from_numpy(X), torch.from_numpy(Y)

class TrainDataset(data.Dataset):

    def __init__(self, path, transform=None):
        self.filenames = get_file_names(path)
        self.transform = transform
        self.fds = None

        self.idx = {}
        self.size = 0
        fds = [h5py.File(f, 'r', libver='latest', smwr=True) for f in self.filenames]

        for i, f in enumerate(fds):
            groups = list(f.keys())

            if 'info' in groups:
                groups.remove('info')
            if 'contigs' in groups:
                groups.remove('contigs')

            for g in groups:
                group_size = f[g].attrs['size']
                for j in range(group_size):
                    self.idx[self.size + j] = (i, g, j)
                self.size += group_size

        for f in fds:
            f.close()

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        file_idx, g, offset = self.idx[idx]

        if not self.fds:
            self.fds = [h5py.File(f, 'r', libver='latest', smwr=True) for f in self.filenames]

        f = self.fds[file_idx]
        group = f[g]

        sample = (group['examples'][offset], group['labels'][offset])
        if self.transform:
            sample = self.transform(sample)

        return sample

class InMemoryTrainDataset(data.Dataset):

    def __init__(self, path, transform=None):
        file_names = get_file_names(path)
        self.transform = transform

        self.X = []
        self.Y = []

        for file_name in file_names:
            with h5py.File(file_name, 'r') as f:

                groups = list(f.keys())
                if 'info' in groups:
                    groups.remove('info')
                if 'contigs' in groups:
                    groups.remove('contigs')

                for group in groups:
                    X = f[group]['examples'][()]
                    Y = f[group]['labels'][()]

                    self.X.extend(list(X))
                    self.Y.extend(list(Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = (self.X[idx], self.Y[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

def get_file_names(path):
    if not os.path.isdir(path): return [path]

    file_names = []
    for f in os.listdir(path):
        if not f.endswith(".hdf5"): continue
        file_names.append(os.path.join(path, f))

    return file_names
