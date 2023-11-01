from pathlib import Path
from typing import Iterable, Tuple, List

import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

from qa_utils.lightning.datasets import PairwiseTrainDatasetBase, PointwiseTrainDatasetBase, ValTestDatasetBase


BertInput = Tuple[List[int], List[int], torch.IntTensor]
BertTrainInput = Tuple[BertInput, BertInput]
BertBatch = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
BertTrainBatch = Tuple[BertBatch, BertBatch]
BertPointwiseTrainInput = Tuple[BertInput, int]
BertPointwiseTrainBatch = Tuple[BertBatch, torch.FloatTensor]
BertPairwiseTrainInput = Tuple[BertInput, BertInput]
BertPairwiseTrainBatch = Tuple[BertBatch, BertBatch]
BertValTestInput = Tuple[int, int, BertInput, int]
BertValTestBatch = Tuple[torch.IntTensor, torch.IntTensor, BertBatch, torch.IntTensor]


def _get_single_bert_input(query: str, doc: str, tokenizer: BertTokenizer) -> BertInput:
    """Tokenize a single (query, document) pair for BERT and compute sentence lengths.

    Args:
        query (str): The query
        doc (str): The document
        tokenizer (BertTokenizer): The tokenizer

    Returns:
        BertInput: Tokenized query and document and document sentence lengths
    """
    # empty queries or documents might cause problems later on
    if len(query.strip()) == 0:
        query = '(empty)'
    if len(doc.strip()) == 0:
        doc = '(empty)'

    query_tokenized = tokenizer.tokenize(query)
    doc_tokenized = []
    sentence_lengths = []
    # BERT has a limit of 512 tokens, we assume that the first 5000 characters include those
    for sentence in nltk.sent_tokenize(doc[:5000]):
        sentence_tokenized = tokenizer.tokenize(sentence)
        doc_tokenized.extend(sentence_tokenized)
        sentence_lengths.append(len(sentence_tokenized))
    return query_tokenized, doc_tokenized, torch.IntTensor(sentence_lengths)


# BERT


def _collate_bert(inputs: Iterable[BertInput], tokenizer: BertTokenizer) -> BertBatch:
    """Collate a number of single BERT inputs, adding special tokens and padding.

    Args:
        inputs (Iterable[BertInput]): The inputs
        tokenizer (BertTokenizer): Tokenizer

    Returns:
        BertBatch: Input IDs, attention masks, token type IDs
    """
    queries_tokenized, docs_tokenized, _ = zip(*inputs)
    inputs = tokenizer(queries_tokenized, docs_tokenized, padding=True, truncation=True)
    return torch.LongTensor(inputs['input_ids']), \
           torch.LongTensor(inputs['attention_mask']), \
           torch.LongTensor(inputs['token_type_ids'])


class BertPointwiseTrainDataset(PointwiseTrainDatasetBase):
    """Dataset for pointwise BERT training.

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, train_file: Path, bert_type: str):
        super().__init__(data_file, train_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

    def get_single_input(self, query: str, doc: str) -> BertInput:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            BertInput: The model input
        """
        return _get_single_bert_input(query, doc, self.tokenizer)

    def collate_fn(self, inputs: Iterable[BertPointwiseTrainInput]) -> BertPointwiseTrainBatch:
        """Collate a number of pointwise inputs.

        Args:
            inputs (Iterable[BertPointwiseTrainInput]): The inputs

        Returns:
            BertPointwiseTrainBatch: A batch of pointwise inputs
        """
        inputs_, labels = zip(*inputs)
        return _collate_bert(inputs_, self.tokenizer), torch.FloatTensor(labels)


class BertPairwiseTrainDataset(PairwiseTrainDatasetBase):
    """Dataset for pairwise BERT training.

    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, train_file: Path, bert_type: str):
        super().__init__(data_file, train_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

    def get_single_input(self, query: str, doc: str) -> BertInput:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            BertInput: The model input
        """
        return _get_single_bert_input(query, doc, self.tokenizer)

    def collate_fn(self, inputs: Iterable[BertPairwiseTrainInput]) -> BertPairwiseTrainBatch:
        """Collate a number of pairwise inputs.

        Args:
            inputs (Iterable[BertPairwiseTrainInput]): The inputs

        Returns:
            BertPairwiseTrainBatch: A batch of pairwise inputs
        """
        pos_inputs, neg_inputs = zip(*inputs)
        return _collate_bert(pos_inputs, self.tokenizer), _collate_bert(neg_inputs, self.tokenizer)


class BertValTestDataset(ValTestDatasetBase):
    """Dataset for BERT validation/testing.

    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validationset/testset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, val_test_file: Path, bert_type: Path):
        super().__init__(data_file, val_test_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

    def get_single_input(self, query: str, doc: str) -> BertInput:
        """Create a single model input from a query and a document.

        Args:
            query (str): The query
            doc (str): The document

        Returns:
            BertInput: The model input
        """
        return _get_single_bert_input(query, doc, self.tokenizer)

    def collate_fn(self, val_test_inputs: Iterable[BertValTestInput]) -> BertValTestBatch:
        """Collate a number of validation/testing inputs.

        Args:
            val_test_inputs (Iterable[BertValInput]): The inputs

        Returns:
            BertValTestBatch: A batch of validation inputs
        """
        q_ids, doc_ids, inputs, labels = zip(*val_test_inputs)
        return torch.IntTensor(q_ids), \
               torch.IntTensor(doc_ids), \
               _collate_bert(inputs, self.tokenizer), \
               torch.IntTensor(labels)
