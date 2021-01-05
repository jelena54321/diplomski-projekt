from torch.utils import data
import os
import h5py
import numpy as np
from torch import _C as torch 

class ToTensor:

    def __call__(self, sample):
        X, Y = sample    
        return torch.from_numpy(X), torch.from_numpy(Y)

class TrainDataset(data.Dataset):

    def __init__(self, path, transform=None):
        self.filenames = get_file_names(path)
        self.transform = transform
        self.fs = None
        self.idx = {}
        self.size = 0
        self.contigs = {}
        
        fs = [h5py.File(f, 'r', libver='latest', smwr=True) for f in self.filenames]
        for i, f in enumerate(fs):
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

            end_group = f['contigs']
            for k in end_group:
                contig = str(k)
                seq = end_group[k].attrs['seq']
                length = end_group[k].attrs['len']

                self.contigs[contig] = (seq, length)

        for f in fs:
            f.close()

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        file_idx, g, offset = self.idx[idx]

        if not self.fs:
            self.fs = [h5py.File(f, 'r', libver='latest', smwr=True) for f in self.filenames]

        f = self.fs[file_idx]
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

                for g in groups:
                    X = f[g]['examples'][()]
                    Y = f[g]['labels'][()]

                    self.X.extend(list(X))
                    self.Y.extend(list(Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = (self.X[idx], self.Y[idx])

        if self.transform: 
            sample = self.transform(sample)

        return sample

class InferenceDataset(data.Dataset):
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.size = 0
        self.idx = {}
        self.f = None
        
        with h5py.File(data_path, 'r') as f:

            groups = list(f.keys())
            groups.remove('contigs')

            for g in groups:
                group_size = f[g].attrs['size']

                for j in range(group_size):
                    self.idx[self.size + j] = (g, j)

                self.size += group_size

    def __getitem__(self, idx):
        if not self.f:
            self.f = h5py.File(self.data_path, 'r')

        g, p = self.idx[idx]
        group = self.f[g]

        contig = group.attrs['contig']
        X = group['examples'][p]
        position = group['position'][p]

        sample = (contig, position, X)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.size

def get_file_names(path):
    if not os.path.isdir(path): return [path]

    file_names = []
    for f in os.listdir(path):
        if not f.endswith(".hdf5"): continue
        file_names.append(os.path.join(path, f))

    return file_names
