import argparse
from data_module import DataModule
from model import RNN
from dataset import InferenceDataset
from torch.utils.data import DataLoader
import torch
from coder import Coder
from collections import defaultdict, Counter
import itertools
from Bio.Seq import Seq
from Bio import SeqIO, SeqRecord

def inference(args):
    cuda_available = torch.cuda.is_available()
    model = RNN.load_from_checkpoint(args.model_path).to('cuda:0' if cuda_available else 'cpu')

    dataset = InferenceDataset(args.data_path)
    dataloader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)

    result = defaultdict(lambda: defaultdict(lambda: Counter()))

    print('>> started inference')
    for batch in dataloader:
        contig, position, X = batch
        X = X.type(torch.cuda.LongTensor if cuda_available else torch.LongTensor)

        output = model(X)
        Y = torch.argmax(output, dim=2).long().cpu().numpy()

        for c, pos, ys in zip(contig, position, Y):
            for p, y in zip(pos, ys):
                base = Coder.decode(y)

                current_position = (p[0].item(), p[1].item())
                result[c][current_position][base] += 1

    print('>> started processing of results')
    contigs = dataset.contigs
    records = []
    for contig in result:
        values = result[contig]

        sorted_positions = sorted(values)
        sorted_positions = list(itertools.dropwhile(lambda x: x[1] != 0, sorted_positions))

        first = sorted_positions[0][0]
        contig_data = contigs[contig]
        seq = contig_data[0][:first]

        for _, p in enumerate(sorted_positions):
            base, _ = values[p].most_common(1)[0]
            if base == Coder.GAP: continue
            seq += base

        last_position = sorted_positions[-1][0]
        seq += contig_data[0][last_position+1:]

        seq = Seq(seq)
        record = SeqRecord.SeqRecord(seq, id=contig)
        records.append(record)

    with open(args.out_path, 'w') as f:
        SeqIO.write(records, f, 'fasta')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    inference(args)

if __name__ == '__main__':
    main()
