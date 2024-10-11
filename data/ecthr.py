import ast
import random

from datasets import load_dataset

from data.DataClass import DataClass


class ecthr(DataClass):

    def __init__(self):
        self.data = []
        self.ids = []
        super().__init__('RashidHaddad/ECTHR-PCR')

    def load_data(self):
        temp_data = load_dataset(self.name, split='train')

        self.ids = temp_data['appno']
        self.data = []
        for facts, laws in zip(temp_data['facts'], temp_data['law']):
            self.data.append(" ".join(facts) + " " + " ".join(laws))

    def get_ids(self):
        return self.ids

    def get_data(self):
        return self.data

    def get_gold_data(self):
        dataset = load_dataset(self.name, split='train')
        cases = dataset['appno']
        citations = dataset['citations']

        data = dict()
        for case, citation_list in zip(cases, citations):
            citations = ast.literal_eval(citation_list)
            if len(citations) > 0:
                data[case] = citations

        return data
