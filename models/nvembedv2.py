from sentence_transformers import SentenceTransformer


class Nvembedv2:
    def __init__(self):
        self.name = 'nvidia/NV-Embed-v2'
        self.model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 1

    def add_eos(self, data):
        data = [input_example + self.model.tokenizer.eos_token for input_example in data]
        return data

    def vectorise(self, data):
        embeddings = self.model.encode(self.add_eos(data), show_progress_bar=True,
                                       batch_size=self.batch_size,
                                       normalize_embeddings=True, truncate=True)
        return embeddings

    def get_name(self):
        return self.name
