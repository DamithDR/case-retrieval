import argparse
import json
import os

import numpy as np
import pandas as pd

from data.coliee import coliee
from data.ecthr import ecthr
from data.ilpcr import ilpcr
from data.irled import irled
from data.muser import muser
from util.name_handler import get_data_class, standadise_name
from sentence_transformers import util

from util.similarity import cosine_similarity


def load_embeddings(dataset, model):
    model_path = standadise_name(model)

    embeddings_path = f'embeddings/{model_path}/{dataset}'
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
    q_embed = query_embeddings[case]
    for key, embedding in candidate_embeddings.items():
        if key != case:
            similarity = cosine_similarity(q_embed, embedding)
            similarity_scores.append(similarity)
            candidate_keys.append(key)
    return candidate_keys, similarity_scores


def get_threshold(query_embeddings, candidate_embeddings, eval):
    current_threshold = 0.95

    results_dict = {}
    for case, citations in eval.items():
        keys, similarity = get_similarity(case, query_embeddings, candidate_embeddings)
        results_dict[case] = {'keys': keys, 'similarity': similarity}

    while current_threshold > 0:
        cases = []
        gold = []
        predictions = []
        for case, citations in eval.items():
            cases.append(case)
            gold.append(citations)
            results = results_dict[case]
            for key, score in zip(results['keys'], results['similarity']):
                if score > current_threshold:
                    predictions.append(key)

        # evaluate and remember the highest f1 and threshold
        current_threshold -= 0.05


def run(dataset, model):
    data_class = get_data_class(dataset)
    eval, test = data_class.get_eval_data()
    candidate_embeddings = load_embeddings(dataset, model)
    query_embeddings = dict()
    if dataset == 'ilpcr' or dataset == 'irled':
        query_ids = data_class.get_query_ids()
        query_embeddings = {key: candidate_embeddings.pop(key) for key in query_ids}
    elif dataset == 'coliee' or dataset == 'muser' or dataset == 'ecthr':
        query_embeddings = candidate_embeddings
    threshold = get_threshold(query_embeddings, candidate_embeddings, eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''case vectoriser arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    args = parser.parse_args()
    run(args.dataset, args.model_name)
