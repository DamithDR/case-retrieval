from models.absembed import AbsEmbed
from sentence_transformers import SentenceTransformer


class Nvembedv2(AbsEmbed):
    def __init__(self):
        super().__init__('nvidia/NV-Embed-v2')
        self.model = SentenceTransformer(self.name, trust_remote_code=True)

        self.model.max_seq_length = 1024
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 2

    def vectorise(self, data):
        embeddings = self.model.encode(data, show_progress_bar=True,
                                       batch_size=self.batch_size,
                                       normalize_embeddings=True)
        return embeddings
