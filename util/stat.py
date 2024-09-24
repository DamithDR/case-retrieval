from datasets import load_dataset

dataset = load_dataset('Exploration-Lab/IL-TUR', "pcr", split='test_candidates')

cases = dataset['text']
# passages = [' \n'.join(passage) for case in cases for passage in case]
cases = [' \n'.join(case) for case in cases]
passages = passages[:1]