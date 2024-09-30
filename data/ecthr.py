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
        self.data = (temp_data['facts'] + ' ' + temp_data['law']).tolist()  # using only the facts of the echr cases

    def get_ids(self):
        return self.ids

    def get_data(self):
        return self.data
