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
    recall_at_50_values = []
    recall_at_100_values = []
    recall_at_500_values = []
    for case, citations in test.items():
        gold.append(citations)
        results = results_dict[case]
        values, labels = sort_by_numbers_desc(results['similarity'], results['keys'])
        predictions.append(labels)
        recall_at_50_values.append(recall_at_k(labels, citations, 50))
        recall_at_100_values.append(recall_at_k(labels, citations, 100))
        recall_at_500_values.append(recall_at_k(labels, citations, 500))
    MAP = mean_average_precision(predictions, gold)
    k_50 = np.mean(recall_at_50_values)
    k_100 = np.mean(recall_at_100_values)
    k_500 = np.mean(recall_at_500_values)

    return MAP, k_50, k_100, k_500


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
    MAP, k_50, k_100, k_500 = calculate_metrics(query_embeddings, candidate_embeddings, gold)

    print(
        f'Model : {model} | Dataset : {dataset} | MAP : {MAP} | recall@50 : {k_50} | recall@100 : {k_100} | recall@500 : {k_500}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''case vectoriser arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    args = parser.parse_args()
    run(args.dataset, args.model_name)
