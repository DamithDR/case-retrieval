import os

import jieba as jieba
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from util.metric import recall_at_k, mean_average_precision, f1_at_k, precision_at_k
from util.name_handler import get_data_class
from util.similarity import sort_by_numbers_desc

numbers = [1] + list(range(5, 51, 5)) + [100]


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
    f1_values = {}
    p_values = {}
    r_values = {}
    for case, citations in gold.items():
        test.append(citations)
        results = results_dict[case]
        values, labels = sort_by_numbers_desc(results['scores'], results['keys'])
        predictions.append(labels)
        # for number in numbers:
        #     if number not in f1_values.keys():
        #         f1_values[number] = []
        #         p_values[number] = []
        #         r_values[number] = []
        #     f1_values[number].append(f1_at_k(labels, citations, number))
        #     p_values[number].append(precision_at_k(labels, citations, number))
        #     r_values[number].append(recall_at_k(labels, citations, number))
    MAP = mean_average_precision(predictions, test)
    f1_final = []
    p_final = []
    r_final = []
    # for number in numbers:
        # f1_final.append(np.mean(f1_values[number]).item())
        # p_final.append(np.mean(p_values[number]).item())
        # r_final.append(np.mean(r_values[number]).item())
    print(f'MAP : {MAP}')
    return MAP, f1_final, p_final, r_final


if __name__ == '__main__':
    datasets = ['irled', 'ilpcr', 'muser', 'coliee', 'ecthr']
    for dataset in datasets:
        MAP, f1_final, p_final, r_final = run(dataset)
        MAP = round(MAP, 2)
        f1_final = [str(round(f1, 2)) for f1 in f1_final]
        p_final = [str(round(p, 2)) for p in p_final]
        r_final = [str(round(r, 2)) for r in r_final]

        results_file_name = 'MAP_results.csv'
        if not os.path.exists(results_file_name):
            with open(results_file_name, 'a') as f:
                f.write("Model,Dataset,Metric,MAP")
                for number in numbers:
                    f.write(f',k_{number}')
                f.write('\n')
        with open(results_file_name, 'a') as f:
            f1_results = ",".join(f1_final)
            p_results = ",".join(p_final)
            r_results = ",".join(r_final)
            f.write(f'BM25,{dataset},F1,{MAP},{f1_results}\n')
            f.write(f'BM25,{dataset},Precision,{MAP},{p_results}\n')
            f.write(f'BM25,{dataset},Recall,{MAP},{r_results}\n')
