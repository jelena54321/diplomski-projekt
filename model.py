from torch.nn import functional as F
from pytorch_lightning.core import lightning as pl

class RNN(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # layers definition
        pass

    def forward(self, x):
        # forward through layers
        # return predicted values
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        # calculate loss using loss function
        # return loss
        pass

    def configure_optimizers(self):
        # return optimizer
        pass
