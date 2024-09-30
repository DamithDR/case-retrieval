import json

import pandas as pd

from data.DataClass import DataClass


class muser(DataClass):

    def __init__(self):
        self.data = []
        self.ids = []
        super().__init__('data/files/muser/muser_cases_pool.json')

    def load_data(self):
        with open(self.name, 'r', encoding='utf-8') as file:
            data = json.load(file)

        rows = [{'id': key, 'text': '\n'.join(value['content']['本院查明']) + '\n'.join(value['content']['本院认为'])}
                for
                key, value in data.items()]
        df = pd.DataFrame(rows)
        self.ids = df['id']
        self.data = df['text']

    def get_ids(self):
        return self.ids

    def get_data(self):
        return self.data
