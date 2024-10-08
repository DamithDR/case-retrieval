from abc import abstractmethod

from transformers import AutoTokenizer


class AbsEmbed():
    def __init__(self, name):
        self.name = name
        self.tokeniser = AutoTokenizer.from_pretrained(self.name)

    @abstractmethod
    def vectorise(self, data):
        pass

    def get_name(self):
        return self.name
