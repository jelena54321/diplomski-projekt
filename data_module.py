import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self):
        # obtain and set training, test and evaluation data
        # self.train = ...
        # self.test = ...
        pass

    def prepare_data(self):
        # prepare data if needed
        pass

    def train_dataloader(self):
        # transforms = ...
        # return DataLoader(self.train, batch_size=self.batch_size)
        pass

    def val_dataloader(self):
        # tranforms = ...
        # return DataLoader(self.val, batch_size=self.batch_size)
        pass

    def test_dataloader(self):
        # transforms = ...
        # return DataLoader(self.test, batch_size=self.batch_size)
        pass
