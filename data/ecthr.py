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

    def get_eval_data(self):
        dataset = load_dataset(self.name, split='train')
        cases = dataset['appno']
        citations = dataset['citations']

        data = dict()
        test_filter = dict()
        for case, citation_list in zip(cases, citations):
            data[case] = ast.literal_eval(citation_list)
            if len(citation_list) > 0:
                test_filter[case] = ast.literal_eval(citation_list)

        random.seed(42)
        split_size = int(len(list(test_filter.keys())) / 10)
        selected_keys = random.sample(list(test_filter.keys()), split_size)

        eval = {key: data.pop(key) for key in selected_keys}
        return eval, data
