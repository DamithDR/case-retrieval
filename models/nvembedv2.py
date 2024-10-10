import torch.nn.functional as F
from transformers import AutoModel

from models.absembed import AbsEmbed


class Nvembedv2(AbsEmbed):
    def __init__(self):
        super().__init__('nvidia/NV-Embed-v2')
        self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).to('cuda')

        self.max_seq_length = 3072
        self.model.tokenizer.padding_side = "right"
        self.batch_size = 8

    def vectorise(self, data):
        all_embeddings = []
        for i in range(0, len(data), self.batch_size):
            if i + self.batch_size > len(data):
                batch_data = data[i:]
            else:
                batch_data = data[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch_data, instruction="", max_length=self.max_seq_length)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings)
        return all_embeddings
