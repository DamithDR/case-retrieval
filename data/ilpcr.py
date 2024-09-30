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
