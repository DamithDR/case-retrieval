from abc import abstractmethod


class DataClass:
    def __init__(self):
        pass

    @abstractmethod
    def vectorise_candidates(self, model_name):
        pass
