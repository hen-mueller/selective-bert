import os
import sys
import pickle
import random
import time
from typing import Dict, List, Optional, Iterable, Tuple, Union
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.utils import logging

import datasets
from datasets import load_dataset, concatenate_datasets, total_allocated_bytes

from transformers import DataCollatorForLanguageModeling

import nltk

class HFDatasetForNextSentencePrediction(Dataset):
    """
    Modified version of TextDatasetForNextSentencePrediction
    https://github.com/huggingface/transformers/blob/de4d7b004a24e4bb087eb46d742ea7939bc74644/src/transformers/data/datasets/language_modeling.py#L258

    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        split_type: str,
        split_perc: int,
        tokenizer: PreTrainedTokenizer,
        save_dir: str,
        bert_type: str,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
        num_workers=None,
    ):
        self.block_size = tokenizer.max_model_input_sizes[bert_type] - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        if split_type == 'train':
            wiki = load_dataset("wikipedia", "20200501.en", split=f"train[:{split_perc}%]")
            bookcorpus = load_dataset("bookcorpusopen", split=f"train[:{split_perc}%]")
        elif split_type == 'val':
            wiki = load_dataset("wikipedia", "20200501.en", split=f"train[{split_perc}%:]")
            bookcorpus = load_dataset("bookcorpusopen", split=f"train[{split_perc}%:]")

        wiki.remove_columns_("title")
        bookcorpus.remove_columns_("title")

        #datasets.logging.set_verbosity_info() #for debugging

        print(f"Preparing the wikipedia dataset", file=sys.stderr)
        wiki = wiki.map(lambda x: self.sentence_split(x), batched=True, num_proc=self.num_workers)

        print(f"Preparing the bookcorpus dataset", file=sys.stderr)
        bookcorpus = bookcorpus.map(lambda x: self.sentence_split(x), batched=True, num_proc=self.num_workers, batch_size=50)

        assert bookcorpus.features.type == wiki.features.type
        bert_dataset = concatenate_datasets([wiki, bookcorpus])

        self.ds = bert_dataset

        print(f"Creating wikipedia nsp examples", file=sys.stderr)
        wiki_nsp = wiki.map(lambda x, i: self.create_examples_from_document(x['text'], i), batched=True, remove_columns=['text'], with_indices=True, num_proc=self.num_workers)#, writer_batch_size=20)#, num_proc=2)
        print(f"Creating bookcorpus nsp examples", file=sys.stderr)
        bookcorpus_nsp = bookcorpus.map(lambda x, i: self.create_examples_from_document(x['text'], i), batched=True, remove_columns=['text'], with_indices=True, num_proc=self.num_workers, batch_size=5)#, writer_batch_size=5)

        print(f"Concat datasets", file=sys.stderr)
        assert bookcorpus.features.type == wiki.features.type
        self.ds = concatenate_datasets([wiki_nsp, bookcorpus_nsp])

    def sentence_split(self, document: Dict[str, List]) -> Dict[str, List]:
        #nltk.download('punkt')
        sentences = {'text':[]}
        for doc in document['text']:
            tokenized_list = []
            sentence_list = nltk.tokenize.sent_tokenize(doc)
            sentence_list = [s.strip("\n ") for s in sentence_list]

            sentence_list = list(filter(None, sentence_list))
            tokenized_list = [self.tokenizer.tokenize(sentence) for sentence in sentence_list]
        
            sentences['text'].append(tokenized_list)
        return sentences

    def create_examples_from_document(self, documents: List[List[List[str]]], doc_index: int):
        """Creates examples for a single documents."""
        max_num_tokens = self.block_size

        examples = {'tokens_a': [], 'tokens_b': [], "is_random_next": []}

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.

        for document in documents:
            target_seq_length = max_num_tokens
            if random.random() < self.short_seq_probability:
                target_seq_length = random.randint(2, max_num_tokens)

            current_length = 0
            i = 0
            chunk_start = 0
            while i < len(document):
                segment = document[i]
                current_length += len(segment)

                if i == len(document) - 1 or current_length >= target_seq_length:
                    if current_length:# and len(document[chunk_start]) < max_num_tokens:
                        # `a_end` is how many segments from `current_chunk` go into the `A`
                        # (first) sentence.
                        a_end = 1
                        len_cur_chunk = i - chunk_start

                        if len_cur_chunk >= 2:
                            a_end = random.randint(1, len_cur_chunk - 1)

                        tokens_a = []

                        for j in range(chunk_start, chunk_start + a_end):
                            tokens_a.extend(document[j])

                        tokens_b = []

                        if len_cur_chunk <= 1 or random.random() < self.nsp_probability:
                            is_random_next = True
                            target_b_length = target_seq_length - len(tokens_a)

                            # This should rarely go for more than one iteration for large
                            # corpora. However, just to be careful, we try to make sure that
                            # the random document is not the same as the document
                            # we're processing.
                            for _ in range(10):
                                random_document_index = random.randint(0, len(self.ds) - 1)
                                if random_document_index != doc_index:
                                    break

                            len_random_doc = len(self.ds[random_document_index]['text'])
                            random_start = random.randint(0, len_random_doc - 1)
                            for j in range(random_start, len_random_doc):
                                tokens_b.extend(self.ds[random_document_index]['text'][j])
                                if len(tokens_b) >= target_b_length:
                                    break
                            # We didn't actually use these segments so we "put them back" so
                            # they don't go to waste.
                            num_unused_segments = len_cur_chunk - a_end
                            i -= num_unused_segments
                        # Actual next
                        else:
                            is_random_next = False
                            for j in range(a_end + chunk_start, i):
                                tokens_b.extend(document[j])

                        assert len(tokens_a) >= 1
                        assert len(tokens_b) >= 1

                        examples['tokens_a'].append(tokens_a)
                        tokens_a = []
                        examples['tokens_b'].append(tokens_b)
                        tokens_b = []
                        examples['is_random_next'].append(is_random_next)

                        target_seq_length = max_num_tokens
                        if random.random() < self.short_seq_probability:
                            target_seq_length = random.randint(2, max_num_tokens)

                    #current_chunk = []
                    current_length = 0
                    chunk_start = i
                    

                i += 1
        return examples

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]

class PretrainingDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm: bool = True, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        #self.device = device
        #super.__init__()

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, List[int]]]]
    ) -> Dict[str, torch.Tensor]:
        batch = {}
        tokens_a = [entry["tokens_a"] for entry in examples]
        tokens_b = [entry["tokens_b"] for entry in examples]

        is_random_next = [entry["is_random_next"] for entry in examples]
        is_random_next = torch.LongTensor(is_random_next)#, device=self.device)

        batch = self.tokenizer(tokens_a, tokens_b, padding=True, truncation=True, return_special_tokens_mask=True, return_tensors="pt")
        batch["next_sentence_label"] = is_random_next

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return (batch['input_ids'], batch['attention_mask'], batch['token_type_ids']), batch['labels'], batch ['next_sentence_label']