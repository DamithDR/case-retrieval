from FlagEmbedding import FlagICLModel

from models.absembed import AbsEmbed


class flag(AbsEmbed):

    def __init__(self):
        super().__init__('BAAI/bge-en-icl')
        self.model = FlagICLModel(self.name, query_instruction_for_retrieval="", normalize_embeddings=True)

        self.max_seq_length = 3072
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 16

    def vectorise(self, data):
        return self.model.encode_corpus(data, max_length=self.max_seq_length, batch_size=self.batch_size)
