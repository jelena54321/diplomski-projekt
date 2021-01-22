import h5py
from temporary_storage import TemporaryTrainStorage, TemporaryInferenceStorage
from abc import ABC
from abc import abstractmethod

class HDF5Writer(ABC):
    """
    A class that represents a data writer for .hdf5 files.

    Attributes
    ----------
    output_path : a path to output .hdf5 file
    train : a flag that indicates whether data is intended for training
    """

    def __init__(self, output_path):
        """
        Parameters
        ----------
        output_path : a path to output .hdf5 file
        """

        self.output_path = output_path
        self.storages = dict()

    def __enter__(self):
        self.f = h5py.File(self.output_path, 'w')
        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def write_contigs(self, refs):
        """
        Writes contigs in the .hdf5 file.

        Parameters
        ----------
        refs : an array of reference sequences
        """

        contigs_group = self.f.create_group('contigs')

        for ref_name, ref in refs:
            contig = contigs_group.create_group(ref_name)
            contig.attrs['name'] = ref_name
            contig.attrs['seq'] = ref
            contig.attrs['len'] = len(ref)

    @abstractmethod
    def store(self, args):
        """
        Stores new data in the temporary storage.
        """
        pass


    def write(self):
        """
        Writes all stored data in the .hdf5 file.
        """

        for storage in self.storages.values():
            self.__write(storage)
            storage.clear()

    def __write(self, storage):
        """
        Writes a single storage chunk in the .hd5f file.
        """

        positions = storage.get_positions()
        if len(positions) == 0: return

        X = storage.get_X()
        Y = storage.get_Y()

        if Y: assert len(positions) == len(X) == len(Y)
        else: assert len(positions) == len(X)

        start, end = positions[0][0][0], positions[-1][-1][0]

        group = self.f.create_group(f'{storage.name}_{start}-{end}')
        group['positions'] = positions

        if Y: group['labels'] = Y

        group.attrs['contig'] = storage.name
        group.attrs['size'] = len(positions)

        group.create_dataset('examples', data=X, chunks=(1, 200, 90))

class InferenceHDF5Writer(HDF5Writer):

    def store(self, args):
        contig, positions, X = args

        if contig in self.storages:
            storage = self.storages[contig]
        else:
            storage = self.storages[contig] = TemporaryInferenceStorage(contig)

        storage.store((positions, X))

class TrainHDF5Writer(HDF5Writer):

    def store(self, args):
        contig, positions, X, Y = args

        if contig in self.storages:
            storage = self.storages[contig]
        else:
            storage = self.storages[contig] = TemporaryTrainStorage(contig)

        storage.store((positions, X, Y))
