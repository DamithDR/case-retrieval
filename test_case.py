import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn import DataParallel
from transformers import AutoTokenizer, AutoModel, pipeline
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



def run(args):
    # Each query needs to be accompanied by an corresponding instruction describing the task.
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question", }

    query_prefix = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "
    queries = [
        'are judo throws allowed in wrestling?',
        'how to become a radiology technician in michigan?'
    ]

    # No instruction needed for retrieval passages
    passage_prefix = ""
    passages = [
        "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
        "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
    ]

    dataset = load_dataset('Exploration-Lab/IL-TUR', "pcr", split='test_candidates')

    cases = dataset['text']
    # passages = [' \n'.join(passage) for case in cases for passage in case]
    # passages = [' \n'.join(case) for case in cases]
    #
    # print(f'length of sequence: {len(passages[args.number].split(" "))}')
    # passages = [passages[args.number]]


    tokeniser = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2')
    # tokenised_data = tokeniser(passages, padding=False, truncation=False, return_tensors="pt")
    # tensor_data = torch.tensor(tokenised_data)

    # load model with tokenizer
    # Initialize the process group for distributed training

    # model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', device_map='auto', trust_remote_code=True)
    pipe = pipeline("feature-extraction", framework="pt", model='nvidia/NV-Embed-v2', device_map="auto", trust_remote_code=True,
                    tokenizer=tokeniser)
    features = pipe(passages, return_tensors="pt", batch_size=1)
    print(features)

    # if torch.cuda.is_available():
    #     print('cuda is available shifting data to cuda')
    #     model = model.to("cuda")
    #     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     for module_key, module in model._modules.items():
    #         model._modules[module_key] = DataParallel(module)

    # tokenised_data = tokenised_data.to('cuda')

    # # get the embeddings
    # max_length = 32768
    # query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)
    # passage_embeddings = model.encode(tokenised_data, instruction=passage_prefix, max_length=max_length)

    # # normalize embeddings
    # query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    # passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

    # get the embeddings with DataLoader (spliting the datasets into multiple mini-batches)
    # batch_size=2
    # query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
    # passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length, num_workers=32, return_numpy=True)

    # scores = (query_embeddings @ passage_embeddings.T) * 100
    # print(scores.tolist())
    # [[87.42693328857422, 0.46283677220344543], [0.965264618396759, 86.03721618652344]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''case retrieval arguments''')
    parser.add_argument('--number', type=int, required=True, help='number')
    args = parser.parse_args()
    run(args)
