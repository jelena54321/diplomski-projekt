import pytorch_lightning as pl
import argparse
from model import RNN
from data_module import DataModule
from pytorch_lightning import callbacks
import torch

BATCH_SIZE = 128
MAX_EPOCHS = 50
PATIENCE = 7
N_GPU = 1

def train(args):
    model = RNN()
    data_module = DataModule(args)

    callbacks_list = None
    if args.val_path:
        callbacks_list = []
        callbacks_list.append(callbacks.EarlyStopping(monitor='val_acc', patience=PATIENCE))
        callbacks_list.append(callbacks.ModelCheckpoint(filepath=args.out_path, monitor='val_acc', prefix='rnn'))

    gpus = N_GPU if torch.cuda.is_available() else None
    trainer = pl.Trainer(gpus=gpus, max_epochs=MAX_EPOCHS, callbacks=callbacks_list)

    trainer.fit(model, datamodule=data_module)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str)
    parser = DataModule.add_data_model_specific_args(parser)
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
