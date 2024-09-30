import os

from data.DataClass import DataClass


class irled(DataClass):

    def __init__(self):
        self.candidate_ids = []
        self.query_ids = []
        self.candidates = []
        self.queries = []
        super().__init__('data/files/irled')

    def load_data(self):
        self.load_candidates()
        self.load_queries()

    def load_candidates(self):
        path = os.path.join(self.name, 'candidates')
        self.candidate_ids, self.candidates = self.load_files(path)

    def load_queries(self):
        path = os.path.join(self.name, 'queries')
        self.query_ids, self.queries = self.load_files(path)

    def load_files(self, path):
        ids = []
        data = []
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                file_path = os.path.join(path, filename)
                with open(file_path, 'r', encoding='windows-1252') as file:
                    text = file.read()
                ids.append(os.path.splitext(filename)[0])
                data.append(text)
        return ids, data

    def get_candidates(self):
        return self.candidates

    def get_candidate_ids(self):
        return self.candidate_ids

    def get_queries(self):
        return self.queries

    def get_query_ids(self):
        return self.query_ids
