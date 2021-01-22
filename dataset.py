from torch.utils import data
import os
import h5py
import numpy as np
import torch

class TrainDataset(data.Dataset):
    """
    A class that defines a training dataset. This dataset does not immediately 
    load and store all data in RAM.

    Attributes
    ----------
    file_names : an array containing all .hdf5 files that represent training dataset
    files : an array of file objects containing training dataset
    idx : a dictionary of indices used for obtaining data
    size : data size
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : a path to .hdf5 file or directory containing .hdf5 files 
            that represent training dataset
        """

        self.file_names = get_file_names(path)
        self.files = None
        self.idx = {}
        self.size = 0

        files = [h5py.File(f, 'r', libver='latest', swmr=True) for f in self.file_names]
        for file_idx, f in enumerate(files):

            groups = list(f.keys())
            if 'info' in groups:
                groups.remove('info')
            if 'contigs' in groups:
                groups.remove('contigs')

            for g in groups:
                group_size = f[g].attrs['size']
                for offset in range(group_size):
                    self.idx[self.size + offset] = (file_idx, g, offset)

                self.size += group_size

        for f in files:
            f.close()

    def __len__(self):
        """
        Returns size of a training dataset.

        Returns
        -------
        size : training dataset size
        """

        return self.size

    def __getitem__(self, idx):
        """
        Obtains training data corresponding to the provided index.

        Parameters
        ----------
        idx : index that corresponds to a training data

        Returns
        -------
        sample : examples and labels corresponding the provided index
        """

        file_idx, g, offset = self.idx[idx]

        if not self.files:
            self.files = [h5py.File(f, 'r', libver='latest', swmr=True) for f in self.file_names]

        f = self.files[file_idx]
        group = f[g]

        sample = (group['examples'][offset], group['labels'][offset])

        return sample

class InMemoryTrainDataset(data.Dataset):
    """
    A class that defines a training dataset. This dataset immediately loads and 
    stores all data in RAM.

    Attributes
    ----------
    X : an array of examples
    Y : an array of labels
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : a path to .hdf5 file or directory containing .hdf5 files 
            that represent training dataset
        """

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

        self.size = len(self.X)

    def __len__(self):
        """
        Returns size of a training dataset.

        Returns
        -------
        size : training dataset size
        """

        return self.size

    def __getitem__(self, idx):
        """
        Obtains training data corresponding to the provided index.

        Parameters
        ----------
        idx : index that corresponds to a training data

        Returns
        -------
        sample : examples and labels corresponding the provided index
        """
        sample = (self.X[idx], self.Y[idx])

        return sample

class InferenceDataset(data.Dataset):
    """
    A class that defines an inference dataset. This dataset does not immediately 
    load and store all data in RAM.

    Attributes
    ----------
    path : path to a file containing inference dataset
    size : inference data size
    f : file object containing inference dataset
    idx : a dictionary of indices used for obtaining data
    contigs : a dictionary of contigs
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : path to a file containing inference dataset
        """

        self.path = path
        self.size = 0
        self.f = None
        self.idx = {}
        self.contigs = {}

        with h5py.File(path, 'r') as f:

            groups = list(f.keys())
            groups.remove('contigs')

            for g in groups:
                group_size = f[g].attrs['size']

                for offset in range(group_size):
                    self.idx[self.size + offset] = (g, offset)

                self.size += group_size

            end_group = f['contigs']
            for ref in end_group:
                contig = str(ref)
                seq = end_group[ref].attrs['seq']
                length = end_group[ref].attrs['len']
                self.contigs[contig] = (seq, length)

    def __getitem__(self, idx):
        """
        Obtains inference data corresponding to the provided index.

        Parameters
        ----------
        idx : index that corresponds to an inference data

        Returns
        -------
        sample : inference sample corresponding the provided index
        """

        if not self.f:
            self.f = h5py.File(self.path, 'r')

        g, offset = self.idx[idx]
        group = self.f[g]

        contig = group.attrs['contig']
        X = group['examples'][offset]
        positions = group['positions'][offset]

        return (contig, positions, X)

    def __len__(self):
        """
        Returns size of an inference dataset.

        Returns
        -------
        size : inference dataset size
        """

        return self.size

def get_file_names(path):
    """
    Returns an array of file names ending with .hdf5 that are stored
    at the provided path.

    Parameters
    ----------
    path : path to a .hdf5 file or directory containing .hdf5 files

    Returns
    -------
    file_names : an array of file names ending with .hdf5
    """

    if not os.path.isdir(path): return [path]

    file_names = []
    for f in os.listdir(path):
        if not f.endswith(".hdf5"): continue
        file_names.append(os.path.join(path, f))

    return file_names
