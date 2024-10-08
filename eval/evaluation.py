import argparse
import os

import numpy as np

from util.metric import recall_at_k, mean_average_precision
from util.name_handler import get_data_class, standadise_name, get_embedding_folder
from util.similarity import cosine_similarity, sort_by_numbers_desc


def load_embeddings(dataset, model):
    model_path = standadise_name(model)
    data_path = get_embedding_folder(dataset)

    embeddings_path = f'embeddings/{model_path}/{data_path}'
    embeddings = dict()

    for filename in os.listdir(embeddings_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(embeddings_path, filename)
            embedding = np.load(file_path)
            embeddings[filename.split('.npy')[0]] = embedding
    return embeddings


def get_embeddings_by_id(df, target_id):
    embeddings = df[df['ids'] == target_id]['embeddings'].values
    return embeddings


def get_similarity(case, query_embeddings, candidate_embeddings):
    similarity_scores = []
    candidate_keys = []
    q_embed = query_embeddings[case.replace('/', '-')]
    for key, embedding in candidate_embeddings.items():
        if key != case:
            similarity = cosine_similarity(q_embed, embedding)
            similarity_scores.append(similarity)
            candidate_keys.append(key)
    return candidate_keys, similarity_scores


def calculate_metrics(query_embeddings, candidate_embeddings, test):
    results_dict = {}
    for case, citations in test.items():
        keys, similarity = get_similarity(case, query_embeddings, candidate_embeddings)
        results_dict[case] = {'keys': keys, 'similarity': similarity}

    gold = []
    predictions = []
    f1_1_values = []
    f1_5_values = []
    f1_10_values = []
    f1_15_values = []
    f1_20_values = []
    f1_50_values = []
    f1_100_values = []
    f1_500_values = []
    for case, citations in test.items():
        gold.append(citations)
        results = results_dict[case]
        values, labels = sort_by_numbers_desc(results['similarity'], results['keys'])
        predictions.append(labels)
        f1_1_values.append(recall_at_k(labels, citations, 1))
        f1_5_values.append(recall_at_k(labels, citations, 5))
        f1_10_values.append(recall_at_k(labels, citations, 10))
        f1_15_values.append(recall_at_k(labels, citations, 15))
        f1_20_values.append(recall_at_k(labels, citations, 20))
        f1_50_values.append(recall_at_k(labels, citations, 50))
        f1_100_values.append(recall_at_k(labels, citations, 100))
        f1_500_values.append(recall_at_k(labels, citations, 500))
    MAP = mean_average_precision(predictions, gold)
    k_1 = np.mean(f1_1_values)
    k_5 = np.mean(f1_5_values)
    k_10 = np.mean(f1_10_values)
    k_15 = np.mean(f1_15_values)
    k_20 = np.mean(f1_20_values)
    k_50 = np.mean(f1_50_values)
    k_100 = np.mean(f1_100_values)
    k_500 = np.mean(f1_500_values)

    return [MAP, k_1, k_5, k_10, k_15, k_20, k_50, k_100, k_500]


def run(dataset, model):
    data_class = get_data_class(dataset)
    gold = data_class.get_gold_data()
    candidate_embeddings = load_embeddings(dataset, model)
    query_embeddings = dict()
    if dataset == 'ilpcr' or dataset == 'irled':
        query_ids = data_class.get_query_ids()
        query_embeddings = {key: candidate_embeddings.pop(key) for key in query_ids}
    elif dataset == 'coliee' or dataset == 'muser' or dataset == 'ecthr':
        query_embeddings = candidate_embeddings
    results = [MAP, k_1, k_5, k_10, k_15, k_20, k_50, k_100, k_500] = calculate_metrics(query_embeddings,
                                                                                        candidate_embeddings, gold)
    results = [str(round(result, 2)) for result in results]

    if not os.path.exists('results.csv'):
        with open('results.csv', 'a') as f:
            f.write("Model,Dataset,MAP,k_1,k_5,k_10,k_15,k_20,k_50,k_100,k_500\n")
    with open('results.csv', 'a') as f:
        results_str = ",".join(results)
        f.write(f'{model},{dataset},{results_str}\n')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='''case vectoriser arguments''')
    # parser.add_argument('--model_name', type=str, required=True, help='model_name')
    # parser.add_argument('--dataset', type=str, required=True, help='dataset')
    # args = parser.parse_args()

    datasets = ['ilpcr', 'coliee', 'irled', 'muser']
    models = ['BAAI/bge-en-icl', 'Salesforce/SFR-Embedding-2_R', 'dunzhang/stella_en_1.5B_v5']

    for model in models:
        for dataset in datasets:
            run(dataset, model)
