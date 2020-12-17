import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse

class DataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.train_path = args.train_path
        self.val_path = args.val_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.is_data_stored_in_RAM = args.memory

    def setup(self):
        # obtain and set training, test and evaluation data
        # self.train = ...
        # self.test = ...
        pass

    def train_dataloader(self):
        # transforms = ...
        # return DataLoader(self.train, batch_size=self.batch_size)
        pass

    def val_dataloader(self):
        # tranforms = ...
        # return DataLoader(self.val, batch_size=self.batch_size)
        pass

    @staticmethod
    def add_data_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('train_path', type=str)
        parser.add_argument('--val_path', type=str, default=None)
        parser.add_argument('--memory', type=str, default=False)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=0)
        return parser
