import argparse
from data_module import DataModule
from model import RNN
from dataset import InferenceDataset, ToTensor
from torch.utils.data import DataLoader
import torch

def inference(args):
    model = RNN.load_from_checkpoint(args.model_path)

    dataset = InferenceDataset(args.data_path, transform=ToTensor())
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    decoding = []
    result = []

    for i, batch in enumerate(dataloader):
        contig, position, X = batch
        # x = x.type(torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor)

        logits = model(X)
        Y = torch.argmax(logits, dim=2).long()
        # Y = Y.cpu().numpy()

        for c, pos, ys in zip(contig, position, Y):
            for p, y in zip(pos, ys):
                base = decoding[y]

                current_position = (p[0].item(), p[1].item())
                result[c][current_position][base] += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    inference(args)

if __name__ == "__main__":
    main()