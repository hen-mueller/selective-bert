# Code to train a selective Bert model from scratch.

The configuration for the used conda environment can be found in the environment.yml file.
```
conda env create -f environment.yml
```

## Data preprocessing
```
python preprocess_pretrain.py DATA_DIR --num_workers 16
```
Open-source replication of the original dataset used to train the selective BERT model. Cleaned and correctly formatted data will be saved in DATA_DIR for further use with out code base.
See `python preprocess_pretrain.py -h` for additional information on the (hyper-)parameters.

## Training
```
python pretrain_selective_bert.py DATA_DIR --save_dir ./out/
```
Train a selective BERT model on the data in DATA_DIR. Use `--gpus` to specify the IDs of the GPUs to be used. Adapt `--batch_size` and `--accumulate_grad_batches` depending on your hardware to match the effective batchsize of 256 from the original BERT paper. Training one episode over the complete dataset has shown to be enough empirically.
See `python pretrain_selective_bert.py -h` for additional information on the (hyper-)parameters.
