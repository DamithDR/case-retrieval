from sentence_transformers import SentenceTransformer
from torch.nn import DataParallel
from transformers import AutoTokenizer


class Nvembedv2:
    def __init__(self):
        self.name = 'nvidia/NV-Embed-v2'
        self.model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
        self.tokeniser = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2')
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 1
        self.max_seq_length = 16 - 1  # keep space for EOS token added in add_eos

    def add_eos(self, data):
        data = [input_example + self.tokeniser.eos_token for input_example in data]
        return data

    def truncate_sequences(self, data):
        truncated_sequences = []
        for seq in data:
            tokens = self.tokeniser.tokenize(seq)
            tokens = tokens[:self.max_seq_length]
            truncated_seq = self.tokeniser.convert_tokens_to_string(tokens)
            truncated_sequences.append(truncated_seq)
        return truncated_sequences

    def vectorise(self, data):
        data = self.truncate_sequences(data)
        embeddings = self.model.encode(self.add_eos(data), show_progress_bar=True,
                                       batch_size=self.batch_size,
                                       normalize_embeddings=True)
        return embeddings

    def get_name(self):
        return self.name
