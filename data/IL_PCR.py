import os.path

import pandas as pd
from torch.nn import DataParallel
from transformers import AutoModel, pipeline, AutoTokenizer

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

    def vectorise_candidates(self, model_name=None):
        self.load_candidates(self.dataset)
        candidates = self.candidates['text']
        candidates = [' \n'.join(candidate) for candidate in candidates]
        # candidates = [candidate[1] for candidate in candidates] #todo change after testing
        candidates = candidates[:100] #todo remove after testing
        tokeniser = AutoTokenizer.from_pretrained(model_name)

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            print(f'visible devices {num_devices}')
        pipe = pipeline("feature-extraction", framework="pt", model=model_name,device_map="auto",trust_remote_code=True,tokenizer=tokeniser)

        print(candidates)
        features = pipe(candidates, return_tensors="pt", batch_size=1)
        print(features)

        # candidate_embeddings = embedding_model.encode(candidates, instruction="", max_length=self.max_length)



        # embeddings_df = pd.DataFrame(candidate_embeddings)
        # model_alias = model_name.split('/')[-1] if model_name.__contains__('/') else model_name
        # dataset_alias = self.dataset.split('/')[-1] if self.dataset.__contains__('/') else self.dataset
        # save_path = f'embeddings/{model_alias}/{dataset_alias}.csv'
        # os.makedirs(save_path)
        # embeddings_df.to_csv(save_path, index=False)
