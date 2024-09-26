from datasets import load_dataset

from data.DataClass import DataClass


class IL_PCR(DataClass):

    def __init__(self):
        self.name = 'Exploration-Lab/IL-TUR'
        super().__init__()

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
