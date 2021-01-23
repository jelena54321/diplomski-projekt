import pytorch_lightning as pl
import argparse
from torch.utils.data import DataLoader
from dataset import InMemoryTrainDataset, TrainDataset
import torch

class DataModule(pl.LightningDataModule):
    """
    A class that manages data used for training. This class obtains data and returns it
    in the required format.

    Attributes
    ----------
    train_path : file path to data used for training
    val_path : file path to data used for validation
    batch_size : size of a single batch
    num_workers : number of subprocesses used for data loading
    is_data_stored_in_RAM : flag that indicates whether all data is immediately loaded and stored in RAM
    """

    def __init__(self, args):
        """
        Parameters
        ----------
        args : an object holding all required arguments
        """

        super().__init__()
        self.train_path = args.train_path
        self.val_path = args.val_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.is_data_stored_in_RAM = args.memory

        self.train = None
        self.val = None

    def setup(self, stage=None):
        """
        Sets up training and validation (if provided) dataset according to
        flag `is_data_stored_in_RAM`.
        """

        data_class = InMemoryTrainDataset if self.is_data_stored_in_RAM else TrainDataset

        self.train = data_class(self.train_path)

        if self.val_path:
            self.val = data_class(self.val_path)

    def train_dataloader(self):
        """
        Returns training data as a `DataLoader` object.

        Returns
        -------
        dataloader : training data
        """

        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns validation data as a `torch.utils.data.DataLoader` object.

        Returns
        -------
        dataloader : validation data

        Raises
        ------
        AssertionError
            If validation is not required, i.e. if val_path is not provided.
        """

        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers) if self.val else None

    @staticmethod
    def add_data_model_specific_args(parent_parser):
        """
        Configures provided parsers by adding data model specific arguments.

        Parameters
        ----------
        parent_parser : parser used for argument configuration

        Returns
        -------
        parser : configured parser
        """

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str, default=None)
        parser.add_argument('--memory', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=0)
        return parser
