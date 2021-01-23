# Project: Consensus polisher implementation using *Pythorch-Lightning* library

This consensus polisher is based on the already existing model [Roko](https://github.com/lbcb-sci/roko) with the exception of using *Pytorch-Lightning* library for a more scientific approach.

## Installation

### GPU
```
git clone https://github.com/jelena54321/diplomski-projekt.git
cd diplomski-projekt
make gpu
```

### CPU
```
git clone https://github.com/jelena54321/diplomski-projekt.git
cd diplomski-projekt
make cpu
```

## Usage

### 1. Activate virtual environment
```
cd diplomski-projekt
. model_venv/bin/activate
```

### 2. Generate features and labels data required for training and inference
```
python generate.py [options ...] --ref_path <reference> --reads_path <reads> --out_path <output>

    --ref_path <str>
        path to a draft assembly in FASTA format
    --reads_path <str> 
        path to reads aligned to the draft assembly in BAM format
    --out_path <str>
        path to an output file in .hdf5 format

    options:
    --truth_genome_path <str>
        path to a truth genome aligned to the draft assembly in BAM format
        (NOTE: required only for generating training data)
    --num_workers <int> 
        default: 1
        number of threads used for data processing
```
Pomoxis [mini_align](https://github.com/nanoporetech/pomoxis/blob/master/scripts/mini_align) tool is recommended for generating BAM files required for data generation.


### 3. Train a model
```
python train.py [options ...] --train_path <train_data> --out_path <model_output>

    --train_path <str>
        path to a directory containing .hdf5 files or a single .hdf5 file
        for training
    --out_path <str>
        path to a directory where model will be saved

    options:
    --val_path <str>
        path to a directory contining .hdf5 files or a single .hdf5 file
        for validation
    --memory <bool>
        default: False
        a flag indicating whether the whole training data will be loaded into RAM for
        training purposes
    --batch_size <int>
        default: 128
        batch size of the training data
    --num_workers <int>
        default: 1
        number of threads used for loading data
```

### 3. Make inference
```
python inference.py [options ...] --model_path <model> --data_path <inference_data> --out_path<output>

    --model_path <str>
        path to a trained model
    --data_path <str>
        path to a .hdf5 file containg inference data
    --out_path <str>
        path to an output file in FASTA format

    options:
    --batch_size <int>
        default: 128
        batch size of the inference data
    --num_workers <int>
        default: 1
        number of threads used for loading data
```
