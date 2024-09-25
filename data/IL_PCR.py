import os.path
import sys

import pandas as pd
from torch.nn import DataParallel
from transformers import AutoModel, pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

from data.DataClass import DataClass
from datasets import load_dataset
import torch


class IL_PCR(DataClass):

    def __init__(self, max_length):
        super().__init__()
        self.queries = None
        self.candidates = None
        self.dataset = 'Exploration-Lab/IL-TUR'
        # self.max_length = 32768
        self.max_length = max_length

    def load_candidates(self, dataset):
        self.candidates = load_dataset(dataset, "pcr", split='test_candidates')

    def load_queries(self, dataset):
        self.candidates = load_dataset(dataset, "pcr", split='test_queries')

    def add_eos(self, input_examples):
        input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    def vectorise_candidates(self, model_name=None):
        self.load_candidates(self.dataset)
        candidates = self.candidates['text']
        ids = self.candidates['id']
        candidates = [' \n'.join(candidate) for candidate in candidates]
        # candidate_embeddings = embedding_model.encode(candidates, instruction="", max_length=self.max_length)

        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"

        # get the embeddings
        batch_size = 2
        candidate_embeddings = self.model.encode(self.add_eos(candidates), show_progress_bar=True,
                                                 batch_size=batch_size,
                                                 normalize_embeddings=True)

        embeddings_df = pd.DataFrame({'ids': ids, 'embeddings': candidate_embeddings})
        model_alias = model_name.split('/')[-1] if model_name.__contains__('/') else model_name
        dataset_alias = self.dataset.split('/')[-1] if self.dataset.__contains__('/') else self.dataset
        save_path = f'embeddings/{model_alias}/{dataset_alias}.csv'
        os.makedirs(save_path)
        embeddings_df.to_csv(save_path, index=False)
