import pytorch_lightning as pl
import argparse
from model import RNN
from data_module import DataModule
from pytorch_lightning import callbacks
import torch

BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 7
N_GPU = 1

def train(args):
    model = RNN()
    data_module = DataModule(args)

    early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=PATIENCE)
    model_checkpoint = callbacks.ModelCheckpoint(filepath=args.out_path, monitor='val_acc', prefix='rnn')
    n_gpus = N_GPU if torch.cuda.is_available() else None
    trainer = pl.Trainer(gpus=n_gpus, max_epochs=MAX_EPOCHS, callbacks=[early_stopping, model_checkpoint])

    trainer.fit(model, datamodule=data_module)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    parser = DataModule.add_data_model_specific_args(parser)
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()