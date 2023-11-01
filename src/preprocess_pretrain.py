#! /usr/bin/env python3

import sys
import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from transformers import DataCollatorForLanguageModeling, BertTokenizerFast
from task_utils.pretrain_datasets import HFDatasetForNextSentencePrediction, PretrainingDataCollatorForLanguageModeling


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('SAVE_DIR', help='Folder for all processed data files')

    # trainer args
    ap.add_argument('--bert_type', default='bert-base-uncased', help='BERT model')

    # remaining args
    ap.add_argument('--random_seed', type=int, default=123, help='Random seed')

    ap.add_argument('--val', action='store_true', help='Create validation split')
    ap.add_argument('--val_split', type=int, default=100, help='Percent of data in training set, (100-val_split) in validation set')

    ap.add_argument('--num_workers', type=int, default=4, help='Number of workers for training DataLoaders')

    args = ap.parse_args()

    # in DDP mode we always need a random seed
    seed_everything(args.random_seed)

    save_dir = Path(args.SAVE_DIR)
    train_path = str(save_dir / 'train')

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_type)

    train_dataset = HFDatasetForNextSentencePrediction('train', args.val_split, tokenizer, args.SAVE_DIR, args.bert_type, num_workers=args.num_workers)
    train_dataset.ds.save_to_disk(train_path)
    if args.val:
        val_path = str(save_dir / 'val')
        val_dataset = HFDatasetForNextSentencePrediction('val', args.val_split, tokenizer, args.SAVE_DIR, args.bert_type, num_workers=args.num_workers)
        val_dataset.ds.save_to_disk(val_path)


    


if __name__ == '__main__':
    main()
