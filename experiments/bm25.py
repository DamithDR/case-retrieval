import jieba as jieba
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from util.metric import recall_at_k, mean_average_precision
from util.name_handler import get_data_class
from util.similarity import sort_by_numbers_desc


def run(dataset):
    nltk.download('punkt')
    data_class = get_data_class(dataset)

    gold = data_class.get_gold_data()

    query_ids = []
    queries = []
    candidate_ids = []
    candidates = []
    if dataset == 'ilpcr' or dataset == 'irled':
        query_ids = data_class.get_query_ids()
        queries = data_class.get_queries()
        candidate_ids = data_class.get_candidate_ids()
        candidates = data_class.get_candidates()
    elif dataset == 'coliee' or dataset == 'muser' or dataset == 'ecthr':
        query_ids = []
        queries = []
        candidate_ids = data_class.get_ids()
        candidates = data_class.get_data()
        for key in gold.keys():
            idx = candidate_ids.index(key)

            num = candidate_ids.pop(idx)  # remove queries from candidates lists
            text = candidates.pop(idx)

            query_ids.append(num)
            queries.append(text)

    # Tokenize the documents
    if dataset == 'muser':
        tokenized_documents = [list(jieba.cut(doc)) for doc in candidates]
    else:
        tokenized_documents = [word_tokenize(doc.lower()) for doc in candidates]
    bm25 = BM25Okapi(tokenized_documents)

    results_dict = {}
    for case, citations in gold.items():
        q_idx = query_ids.index(case)
        query = queries[q_idx]
        if dataset == 'muser':
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = word_tokenize(query.lower())

        scores = bm25.get_scores(tokenized_query)

        results_dict[case] = {'keys': candidate_ids, 'scores': scores}

    test = []
    predictions = []
    recall_at_50_values = []
    recall_at_100_values = []
    recall_at_500_values = []
    for case, citations in gold.items():
        test.append(citations)
        results = results_dict[case]
        values, labels = sort_by_numbers_desc(results['scores'], results['keys'])
        predictions.append(labels)
        recall_at_50_values.append(recall_at_k(labels, citations, 50))
        recall_at_100_values.append(recall_at_k(labels, citations, 100))
        recall_at_500_values.append(recall_at_k(labels, citations, 500))
    MAP = mean_average_precision(predictions, test)
    k_50 = np.mean(recall_at_50_values)
    k_100 = np.mean(recall_at_100_values)
    k_500 = np.mean(recall_at_500_values)
    with open('bm25.txt', 'a') as f:
        f.write(
            f'Model : BM25 | Dataset : {dataset} | MAP : {MAP} | recall@50 : {k_50} | recall@100 : {k_100} | recall@500 : {k_500} \n')


if __name__ == '__main__':
    run('ilpcr')
    run('irled')
    run('coliee')
    # run('muser')
