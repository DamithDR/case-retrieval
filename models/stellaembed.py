from models.absembed import AbsEmbed
from sentence_transformers import SentenceTransformer


class stella(AbsEmbed):

    def __init__(self):
        super().__init__('dunzhang/stella_en_1.5B_v5')
        self.model = SentenceTransformer(self.name, trust_remote_code=True).cuda()

        self.model.max_seq_length = 1024
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 2

    def vectorise(self, data):
        embeddings = self.model.encode(data, show_progress_bar=True,
                                       batch_size=self.batch_size,
                                       normalize_embeddings=True)
        return embeddings
