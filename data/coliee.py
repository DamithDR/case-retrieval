import os

from data.DataClass import DataClass


class coliee(DataClass):

    def __init__(self):
        self.data = []
        self.ids = []
        super().__init__('data/files/coliee')

    def load_data(self):
        for filename in os.listdir(self.name):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.name, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                self.ids.append(os.path.splitext(filename)[0])
                self.data.append(text)

    def get_ids(self):
        return self.ids

    def get_data(self):
        return self.data
