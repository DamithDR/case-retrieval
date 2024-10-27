from models.absembed import AbsEmbed
from sentence_transformers import SentenceTransformer

class bertbase(AbsEmbed):

    def __init__(self):
        super().__init__('google-bert/bert-base-uncased')
        self.model = SentenceTransformer(self.name, trust_remote_code=True)

        self.model.max_seq_length = 512
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 16

    def vectorise(self, data):
        embeddings = self.model.encode(data, show_progress_bar=True,
                                       batch_size=self.batch_size,
                                       normalize_embeddings=True)
        return embeddings