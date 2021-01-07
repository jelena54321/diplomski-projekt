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
    """
    A class that defines a training dataset. This dataset does not immediately 
    load and store all data in RAM.

    Attributes
    ----------
    file_names : str array
        an array containing all .hdf5 files that represent training dataset
    files : `h5py.File` array
        an array of file objects containing training dataset
    idx : int: (int, str, int) dictionary
        a dictionary of indices used for obtaining data
    size : int
        data size
    """

    def __init__(self, path, transform=None):
        """
        Parameters
        ----------
        path : str
            a path to .hdf5 file or directory containing .hdf5 files that represent training dataset
        """

        self.file_names = get_file_names(path)
        self.transform = transform
        self.files = None
        self.idx = {}
        self.size = 0
        
        files = [h5py.File(f, 'r', libver='latest', smwr=True) for f in self.file_names]
        for i, f in enumerate(files):
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

        for f in files:
            f.close()

    def __len__(self):
        """
        Returns size of a training dataset.

        Returns
        -------
        size : int
            training dataset size
        """
        
        return self.size
    
    def __getitem__(self, idx):
        """
        Obtains training data corresponding to the provided index.

        Parameters
        ----------
        idx : int
            index that corresponds to a training data

        Returns
        -------
        sample : (..., ...)
        """

        file_idx, g, offset = self.idx[idx]

        if not self.files:
            self.files = [h5py.File(f, 'r', libver='latest', smwr=True) for f in self.file_names]

        f = self.files[file_idx]
        group = f[g]

        sample = (group['examples'][offset], group['labels'][offset])
        if self.transform:
            sample = self.transform(sample)

        return sample

class InMemoryTrainDataset(data.Dataset):
    """
    A class that defines a training dataset. This dataset immediately loads and 
    stores all data in RAM.

    Attributes
    ----------
    X : ...
        an array of features
    Y : ...
        an array of labels
    """

    def __init__(self, path, transform=None):
        """
        Parameters
        ----------
        path : str
            a path to .hdf5 file or directory containing .hdf5 files that represent training dataset
        """

        self.transform = transform
        self.X = []
        self.Y = []

        for file_name in get_file_names(path):
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
        """
        Returns size of a training dataset.

        Returns
        -------
        size : int
            training dataset size
        """
        
        return len(self.X)

    def __getitem__(self, idx):
        """
        Obtains training data corresponding to the provided index.

        Parameters
        ----------
        idx : int
            index that corresponds to a training data

        Returns
        -------
        sample : (..., ...)
        """
        
        sample = (self.X[idx], self.Y[idx])

        if self.transform: 
            sample = self.transform(sample)

        return sample

class InferenceDataset(data.Dataset):
    """
    A class that defines an inference dataset. This dataset does not immediately 
    load and store all data in RAM.

    Attributes
    ----------
    path : str
        path to a file containing inference dataset
    size : int
        inference data size
    f : `h5py.File`
        file object containing inference dataset
    idx : int: (str, int) dictionary
        a dictionary of indices used for obtaining data
    contigs : str: ..., int
        a dictionary of contigs
    """
    
    def __init__(self, path, transform=None):
        """
        Parameters
        ----------
        path : str
            path to a file containing inference dataset
        """

        self.path = path
        self.transform = transform
        self.size = 0
        self.f = None
        self.idx = {}
        self.contigs = {}
        
        with h5py.File(path, 'r') as f:

            groups = list(f.keys())
            groups.remove('contigs')

            for g in groups:
                group_size = f[g].attrs['size']

                for j in range(group_size):
                    self.idx[self.size + j] = (g, j)

                self.size += group_size

            end_group = f['contigs']
            for k in end_group:
                contig = str(k)
                seq = end_group[k].attrs['seq']
                length = end_group[k].attrs['len']
                self.contigs[contig] = (seq, length)

    def __getitem__(self, idx):
        """
        Obtains inference data corresponding to the provided index.

        Parameters
        ----------
        idx : int
            index that corresponds to an inference data

        Returns
        -------
        sample : (..., ..., ...)
        """

        if not self.f:
            self.f = h5py.File(self.path, 'r')

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
        """
        Returns size of an inference dataset.

        Returns
        -------
        size : int
            inference dataset size
        """
    
        return self.size

def get_file_names(path):
    """
    Returns an array of file names ending with .hdf5 that are stored
    at the provided path.

    Parameters
    ----------
    path : str
        path to a .hdf5 file or directory containing .hdf5 files

    Returns
    -------
    file_names : str array
        an array of file names ending with .hdf5
    """

    if not os.path.isdir(path): return [path]

    file_names = []
    for f in os.listdir(path):
        if not f.endswith(".hdf5"): continue
        file_names.append(os.path.join(path, f))

    return file_names
