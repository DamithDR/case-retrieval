import os.path

import pandas as pd
from torch.nn import DataParallel
from transformers import AutoModel

from data.DataClass import DataClass
from datasets import load_dataset


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

    def vectorise_candidates(self, model_name=None):
        self.load_candidates(self.dataset)
        candidates = self.candidates['text']
        embedding_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for module_key, module in embedding_model._modules.items():
            embedding_model._modules[module_key] = DataParallel(module) #use multiple gpus
        candidate_embeddings = embedding_model.encode(candidates, instruction="", max_length=self.max_length)
        embeddings_df = pd.DataFrame(candidate_embeddings)
        model_alias = model_name.split('/')[-1] if model_name.__contains__('/') else model_name
        dataset_alias = self.dataset.split('/')[-1] if self.dataset.__contains__('/') else self.dataset
        save_path = f'embeddings/{model_alias}/{dataset_alias}.csv'
        os.makedirs(save_path)
        embeddings_df.to_csv(save_path, index=False)
