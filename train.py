import pytorch_lightning as pl
import argparse
from model import *
from data_module import *
from pytorch_lightning import callbacks

BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 7


def train(args):
    model = RNN()
    data_module = DataModule(args)

    early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=PATIENCE)
    model_checkpoint = callbacks.ModelCheckpoint(filepath=args.out, monitor='val_acc', prefix='rnn')
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, callbacks=[early_stopping, model_checkpoint])
    trainer.fit(model, data_module.train_dataloader, [data_module.val_dataloader])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=str)
    parser = DataModule.add_data_model_specific_args(parser)
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()