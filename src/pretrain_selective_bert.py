#! /usr/bin/env python3

import sys
import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from models.selective_bert_pretrain import SelectiveBertForPreTraining

from transformers import DataCollatorForLanguageModeling, BertModel, BertForPreTraining
from task_utils.pretrain_datasets import HFDatasetForNextSentencePrediction, PretrainingDataCollatorForLanguageModeling

from datasets import load_from_disk


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('DATA_DIR', help='Folder for all processed data files')

    # trainer args
    # Trainer.add_argparse_args would make the help too cluttered
    ap.add_argument('--accumulate_grad_batches', type=int, default=1, help='Update weights after this many batches')
    ap.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    ap.add_argument('--gpus', type=int, nargs='+', help='GPU IDs to train on')
    ap.add_argument('--val_check_interval', type=float, default=1.0, help='Validation check interval')
    ap.add_argument('--save_top_k', type=int, default=1, help='Save top-k checkpoints')
    ap.add_argument('--limit_val_batches', type=float, default=sys.maxsize, help='Use a subset of validation data')
    ap.add_argument('--limit_train_batches', type=float, default=sys.maxsize, help='Use a subset of training data')
    ap.add_argument('--limit_test_batches', type=float, default=sys.maxsize, help='Use a subset of test data')
    ap.add_argument('--precision', type=int, choices=[16, 32], default=32, help='Floating point precision')
    ap.add_argument('--accelerator', default='ddp', help='Distributed backend (accelerator)')

    # model args
    BertGumbelForPretraining.add_model_specific_args(ap)

    # remaining args
    #ap.add_argument('--val_patience', type=int, default=3, help='Validation patience')
    ap.add_argument('--save_dir', default='out', help='Directory for logs, checkpoints and predictions')
    ap.add_argument('--random_seed', type=int, default=123, help='Random seed')
    ap.add_argument('--load_weights', help='Load pre-trained weights before training')
    #ap.add_argument('--test', action='store_true', help='Test the model after training')

    #ap.add_argument('--base', action='store_true', help='Use the default BERT model')
    ap.add_argument('--init_bert', action='store_true', help='Initialize the BERT model from huggingface')
    ap.add_argument('--continue_ckpt', help='Path to checkpoint of pretrained BertForPreTraining model')

    args = ap.parse_args()

    # in DDP mode we always need a random seed
    seed_everything(args.random_seed)

    data_dir = Path(args.DATA_DIR)

    train_dir = data_dir / 'train'
    train_dataset = load_from_disk(str(train_dir))

    # uses one batch of training data in validation for val_ds=None to allow use of pytorch_lightning.ModelCheckpoint
    model = SelectiveBertForPreTraining(vars(args), train_ds=train_dataset, val_ds=None)

    if args.load_weights:
        weights = torch.load(args.load_weights)
        model.load_state_dict(weights['state_dict'])
        del weights
        torch.cuda.empty_cache()

    if args.init_bert:
        reference = BertForPreTraining.from_pretrained(args.bert_type)
        #reference.cls.predictions.decoder.bias = torch.nn.Parameter(torch.zeros(reference.config.vocab_size))
        model.bert = reference.bert
        model.cls = reference.cls
        if args.freeze_bert:
            for p in model.bert.parameters():
                p.requires_grad = False
            #for p in model.cls.parameters():
            #    p.requires_grad = False
        del reference
        torch.cuda.empty_cache()

    #if not args.continue_ckpt:
    model_checkpoint = ModelCheckpoint(monitor='val_epoch', mode='max', save_top_k=args.save_top_k, verbose=True)
    #else:
    #    model_checkpoint = ModelCheckpoint(monitor='val_epoch', mode='max', save_top_k=args.save_top_k, verbose=True, dirpath=args.continue_ckpt)

    if not args.continue_ckpt:
        trainer = Trainer.from_argparse_args(args, deterministic=True,
                                         replace_sampler_ddp=False,
                                         default_root_dir=args.save_dir,
                                         checkpoint_callback=model_checkpoint,
                                         callbacks=[LearningRateMonitor()])
    else:
        trainer = Trainer.from_argparse_args(args, deterministic=True,
                                         resume_from_checkpoint=args.continue_ckpt,
                                         replace_sampler_ddp=False,
                                         default_root_dir=args.save_dir,
                                         checkpoint_callback=model_checkpoint,
                                         callbacks=[LearningRateMonitor()])

    trainer.fit(model)
    if args.test:
        trainer.test()
    #trainer.save_checkpoint()


if __name__ == '__main__':
    main()
