import random

from datasets import load_dataset

from data.DataClass import DataClass


class ilpcr(DataClass):

    def __init__(self):
        self.queries = []
        self.query_ids = []
        self.candidates = []
        self.candidates_ids = []
        super().__init__('Exploration-Lab/IL-TUR')


    def load_data(self):
        self.load_candidates()
        self.load_queries()

    def load_candidates(self):
        temp_data = load_dataset(self.name, "pcr", split='test_candidates')
        case_list = temp_data['text']
        self.candidates_ids = temp_data['id']
        self.candidates = [' \n'.join(case) for case in case_list]

    def load_queries(self):
        temp_data = load_dataset(self.name, "pcr", split='test_queries')
        case_list = temp_data['text']
        self.query_ids = temp_data['id']
        self.queries = [' \n'.join(case) for case in case_list]

    def get_candidates(self):
        return self.candidates

    def get_candidate_ids(self):
        return self.candidates_ids

    def get_queries(self):
        return self.queries

    def get_query_ids(self):
        return self.query_ids

    def get_eval_data(self):
        dataset = load_dataset(self.name, "pcr", split='test_queries')
        cases = dataset['id']
        citations = dataset['relevant_candidates']

        data = dict()
        test_filter = dict()
        for case, citation_list in zip(cases, citations):
            data[case] = citation_list
            if len(citation_list) > 0:
                test_filter[case] = citation_list

        random.seed(42)
        split_size = int(len(list(test_filter.keys())) / 10)
        selected_keys = random.sample(list(test_filter.keys()), split_size)

        eval = {key: data.pop(key) for key in selected_keys}
        return eval, data
