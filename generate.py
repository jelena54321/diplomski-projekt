import argparse
from data_generator import generate_inference_data, generate_train_data, generate_regions
from Bio import SeqIO
from hdf5_writer import TrainHDF5Writer, InferenceHDF5Writer
from multiprocessing import Pool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reads_path', type=str)
    parser.add_argument('--truth_genome_path', type=str, default=None)
    parser.add_argument('--ref_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    with open(args.ref_path, 'r') as ref_file:
        refs = [(str(r.id), str(r.seq)) for r in SeqIO.parse(ref_file, 'fasta')]

    train = args.truth_genome_path is not None
    generation_function = generate_train_data if train else generate_inference_data
    data_writer_class = TrainHDF5Writer if train else InferenceHDF5Writer

    with data_writer_class(args.out_path) as writer:
        writer.write_contigs(refs)

        arguments = []
        for ref_name, ref in refs:
            for region in generate_regions(ref, ref_name):
                arguments.append((args.reads_path, args.truth_genome_path, ref, region) if train else (args.reads_path, ref, region))

        print(f'>> data generation started - number of tasks: {len(arguments)}')

        with Pool(processes=args.num_workers) as pool:
            regions_finished = 0
            for result in pool.imap(generation_function, arguments):
                if not result: continue
                writer.store(result)
                regions_finished += 1

                if regions_finished % 10 == 0:
                    print(f'>> writing to disk started')
                    writer.write()
                    print(f'>> writing to disk finished')

            writer.write()

if __name__ == '__main__':
    main()
