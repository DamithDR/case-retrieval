from abc import abstractmethod


class DataClass:
    def __init__(self, name):
        self.name = name

        self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    def get_name(self):
        return self.name
