from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, BertTokenizer, BertTokenizerFast, AdamW, get_constant_schedule_with_warmup, AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertPreTrainingHeads

from models.datasets import BertBatch
from pytorch_lightning import LightningModule

import time

from datasets import Dataset
from task_utils.pretrain_datasets import PretrainingDataCollatorForLanguageModeling

InputBatch = Any
TrainingBatch = Tuple[InputBatch, InputBatch, InputBatch]
ValTestBatch = Tuple[InputBatch, InputBatch, InputBatch]

class SelectiveBertForPreTraining(LightningModule):
    """Experimental BERT ranker. Changed to work like transformers.BertForPreTraining.

    Args:
        hparams (Dict[str, Any]): All model hyperparameters
        num_workers (int, optional): Number of DataLoader workers. Defaults to 16.
    """
    def __init__(self, hparams: Dict[str, Any], train_ds: Dataset = None, val_ds: Dataset = None, test_ds: Dataset = None, num_workers: int = 16):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        #backwards compatibility
        super().__init__()
        if 'distributed_backend' in hparams:
            hparams['accelerator'] = hparams['distributed_backend']
        uses_ddp = 'ddp' in hparams['accelerator']
        self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss_margin = hparams['loss_margin']
        self.batch_size = hparams['batch_size']
        self.num_workers = num_workers
        self.uses_ddp = 'ddp' in hparams['accelerator']

        self.dropout = torch.nn.Dropout(hparams['dropout'])

        #https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForPreTraining
        self.tokenizer = BertTokenizer.from_pretrained(hparams['bert_type'])
        self.config = AutoConfig.from_pretrained(hparams['bert_type'])
        self.bert = AutoModel.from_config(self.config)
        self.cls = BertPreTrainingHeads(self.config)

        self.collator = PretrainingDataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        self.score_list = []
        self.time_list = []

        #3 linear layers
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(hparams['bert_dim'], hparams['bert_dim']//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams['bert_dim']//2, hparams['bert_dim']//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams['bert_dim']//2, hparams['bert_dim']//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hparams['bert_dim']//2, 1),
            torch.nn.Sigmoid()
        )
        

        if 'selector_dropout' in hparams:#backwards compatiblitiy
            if hparams['selector_dropout']:
                self.selector_dropout = torch.nn.Dropout(hparams['selector_dropout'])

        if 'freeze_bert' in hparams:
            if hparams['freeze_bert']:
                for p in self.bert.parameters():
                    p.requires_grad = False

    def train_dataloader(self) -> DataLoader:
        """Return a trainset DataLoader. If the trainset object has a function named `collate_fn`,
        it is used. If the model is trained in DDP mode, the standard `DistributedSampler` is used.

        Returns:
            DataLoader: The DataLoader
        """
        if self.uses_ddp:
            sampler = DistributedSampler(self.train_ds, shuffle=True)
            shuffle = None
        else:
            sampler = None
            shuffle = True

        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, shuffle=shuffle,
                          num_workers=self.num_workers, collate_fn=self.collator)

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a validationset DataLoader if the validationset exists. If the validationset object has a function
        named `collate_fn`, it is used. If the model is validated in DDP mode, `DistributedQuerySampler` is used
        for ranking metrics to work on a query level.

        Returns:
            Optional[DataLoader]: The DataLoader, or None if there is no validation dataset
        """
        if self.val_ds is None:
            sampler = DistributedSampler(self.train_ds.select(range(self.batch_size)))
            return DataLoader(self.train_ds.select(range(self.batch_size)), batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)
            #return None

        if self.uses_ddp:
            sampler = DistributedSampler(self.val_ds)
        else:
            sampler = None

        return DataLoader(self.val_ds, batch_size=self.batch_size, sampler=sampler, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self.collator)

    def forward(self, batch: BertBatch) -> torch.Tensor:
        """Compute the relevance scores for a batch.

        Args:
            batch (BertBatch): BERT inputs

        Returns:
            torch.Tensor: The output scores, shape (batch_size, 1)
        """
        (input_ids, attention_mask, token_type_ids) = batch

        embeddings = self.bert.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)
        
        scores = self.selector(inputs_embeds)

        if 'selector_dropout' in self.hparams:#backwards compatiblitiy
            if self.hparams['selector_dropout']:
                if self.training:
                    clean_scores = scores
                scores = self.selector_dropout(scores)

        if not self.hparams['st']:
            inputs_embeds = inputs_embeds * scores
        else:
            inputs_embeds = inputs_embeds * scores + (inputs_embeds*(1-scores)).detach()

        sequence_output, pooled_output = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        out = (prediction_scores, seq_relationship_score)

        if self.training:
            return out, scores, clean_scores
        return out

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> torch.Tensor:
        """Train a single batch.
    
        Args:
            batch (TrainingBatch): A pairwise training batch (positive and negative inputs)
            batch_idx (int): Batch index
    
        Returns:
            torch.Tensor: Training loss
        """
        scores = []

        inputs, labels, next_sentence_label = batch
        (prediction_scores, seq_relationship_score), scores, clean_scores = self(inputs)

        loss_fct = torch.nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))

        loss = masked_lm_loss + next_sentence_loss \
            + self.hparams['l1']*torch.sum(scores) \
        
        self.log('train_loss', loss)
        clean_scores = torch.masked_select(clean_scores, batch[0][1].unsqueeze(dim=2).bool())
        self.log('train_score_stddev', torch.std(clean_scores))
        #Logs score percentiles every 250 steps
        if self.hparams['log_scores'] and torch.is_tensor(clean_scores) and self.global_step % 250 == 0:
            #filter padding with attention mask
            clean_scores = clean_scores.double()
            self.logger.experiment.add_scalars('score_distribution', 
                                                {'mean': torch.mean(clean_scores), 
                                                'min': torch.min(clean_scores), 
                                                '25perc': torch.quantile(clean_scores, 0.25), 
                                                '50perc': torch.quantile(clean_scores, 0.5), 
                                                '75perc': torch.quantile(clean_scores, 0.75),
                                                'max': torch.max(clean_scores)},
                                                global_step=self.global_step)
        return loss

    def validation_step(self, batch: ValTestBatch, batch_idx: int) -> Tuple[torch.Tensor]:
        """Process a single validation batch.

        Args:
            batch (ValTestBatch): Query IDs, document IDs, inputs and labels
            batch_idx (int): Batch index
        
        Returns:
            Tuple[torch.Tensor]: Query IDs, predictions and labels
        """
        inputs, labels, next_sentence_label = batch

        t0 = time.time()
        outputs = self(inputs)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        runtime = torch.as_tensor(t1 - t0, device=self.device)

        #outputs = (prediction_scores, seq_relationship_score)
        return outputs, labels, next_sentence_label, runtime

    def validation_epoch_end(self, val_results: List[Tuple[torch.Tensor]]):
        """Accumulate all validation batches and compute MAP and MRR@k. The results are approximate in DDP mode.

        Args:
            val_results (List[Tuple[torch.Tensor]]): Query IDs, predictions and labels

        Returns:
            EvalResult: MAP and MRR@k
        """
        times, mlm_loss, nsp_loss = [], [], []
        for (prediction_scores, seq_relationship_score), labels, next_sentence_label, runtime in val_results:
            times.append(runtime)
            
            loss_fct = torch.nn.CrossEntropyLoss()
            #shorter when removing tokens
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))

            mlm_loss.append(masked_lm_loss)
            nsp_loss.append(next_sentence_loss)

        self.log('val_epoch', torch.tensor(self.current_epoch, device=self.device))
        self.log('val_avg_runtime', torch.mean(torch.stack(times)), sync_dist=True, sync_dist_op='mean')
        self.log('val_mlm_loss', torch.mean(torch.stack(mlm_loss)), sync_dist=True, sync_dist_op='mean')
        self.log('val_nsp_loss', torch.mean(torch.stack(nsp_loss)), sync_dist=True, sync_dist_op='mean')

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Create an AdamW optimizer using constant schedule with warmup.

        Returns:
            Tuple[List[Any], List[Any]]: The optimizer and scheduler
        """
        opt = AdamW(self.parameters(), lr=self.hparams['lr'])#, weight_decay=self.hparams['weight_decay'])
        sched = get_constant_schedule_with_warmup(opt, self.hparams['warmup_steps'])
        return [opt], [{'scheduler': sched, 'interval': 'step'}]

    @staticmethod
    def add_model_specific_args(ap: ArgumentParser):
        """Add model-specific arguments to the parser.

        Args:
            ap (ArgumentParser): The parser
        """
        ap.add_argument('--bert_type', default='bert-base-uncased', help='BERT model')
        ap.add_argument('--bert_dim', type=int, default=768, help='BERT output dimension')
        ap.add_argument('--dropout', type=float, default=0.1, help='Dropout percentage')
        ap.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
        ap.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay rate')
        ap.add_argument('--loss_margin', type=float, default=0.2, help='Hinge loss margin')
        ap.add_argument('--batch_size', type=int, default=32, help='Batch size')
        ap.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')

        ap.add_argument('--l1', type=float, default=0.0, help='Lambda for L1 regularization of scores')
        ap.add_argument('--delta', type=float, default=0.5, help='Cutoff value for removing tokens')

        ap.add_argument('--log_scores', action='store_true', help='Log scoring statistics during training')
        ap.add_argument('--st', action='store_true', help='Straight trough optimization during training')
        ap.add_argument('--freeze_bert', action='store_true', help='Freeze BERT, only train selector')
        ap.add_argument('--selector_dropout', type=float, default=0.0, help='Dropout percentage between selector and BERT model')
        return ap