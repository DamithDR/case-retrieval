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
                file_path = os.path.join(self.name, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                ids.append(os.path.splitext(filename)[0])
                data.append(text)
        return ids, text

