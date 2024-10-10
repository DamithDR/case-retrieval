from sentence_transformers import SentenceTransformer

from models.absembed import AbsEmbed


class flag(AbsEmbed):

    def __init__(self):
        super().__init__('BAAI/bge-en-icl')
        self.model = SentenceTransformer(self.name, trust_remote_code=True)

        self.max_seq_length = 3072
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 2

    def vectorise(self, data):
        embeddings = self.model.encode(data, show_progress_bar=True,
                                       batch_size=self.batch_size,
                                       normalize_embeddings=True)
        return embeddings
