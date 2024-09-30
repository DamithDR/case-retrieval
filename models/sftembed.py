from models.absembed import AbsEmbed
from sentence_transformers import SentenceTransformer


class sfr(AbsEmbed):

    def __init__(self):
        super().__init__("Salesforce/SFR-Embedding-2_R")
        self.model = SentenceTransformer(self.name, trust_remote_code=True)

        self.model.max_seq_length = 3072
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 2

    def vectorise(self, data):
        embeddings = self.model.encode(data, show_progress_bar=True,
                                       batch_size=self.batch_size,
                                       normalize_embeddings=True)
        return embeddings