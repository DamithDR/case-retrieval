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
    query_embeddings = None
    candidate_embeddings = None
    if dataset == 'coliee':
        df = pd.read_csv(f'embeddings/{model_path}/coliee_embeddings.csv')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(json.loads(x)))
        # df['embeddings'] = df['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        query_embeddings = candidate_embeddings = df
    elif dataset == 'muser':
        df = pd.read_csv(f'embeddings/{model_path}/muser_cases_pool.json_embeddings.csv')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(json.loads(x)))
        query_embeddings = candidate_embeddings = df
    elif dataset == 'ecthr':
        df = pd.read_csv(f'embeddings/{model_path}/ECTHR-PCR_embeddings.csv')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(json.loads(x)))
        query_embeddings = candidate_embeddings = df
    elif dataset == 'irled':
        query_embeddings = pd.read_csv(f'embeddings/{model_path}/irled_queries.csv')
        query_embeddings['embeddings'] = query_embeddings['embeddings'].apply(lambda x: np.array(json.loads(x)))

        candidate_embeddings = pd.read_csv(f'embeddings/{model_path}/irled_candidates.csv')
        candidate_embeddings['embeddings'] = candidate_embeddings['embeddings'].apply(lambda x: np.array(json.loads(x)))
    elif dataset == 'ilpcr':
        query_embeddings = pd.read_csv(f'embeddings/{model_path}/IL-TUR_queries.csv')
        query_embeddings['embeddings'] = query_embeddings['embeddings'].apply(lambda x: np.array(json.loads(x)))

        candidate_embeddings = pd.read_csv(f'embeddings/{model_path}/IL-TUR_candidates.csv')
        candidate_embeddings['embeddings'] = candidate_embeddings['embeddings'].apply(lambda x: np.array(json.loads(x)))

    return query_embeddings, candidate_embeddings


def get_embeddings_by_id(df, target_id):
    embeddings = df[df['ids'] == target_id]['embeddings'].values
    return embeddings


def get_similarity(query, candidates):
    similarity_scores = []
    for candidate in candidates:
        similarity = cosine_similarity(query, candidate)
        similarity_scores.append(similarity)
    print(similarity_scores)


def get_threshold(query_embeddings, candidate_embeddings, eval):
    for case, citations in eval.items():
        query = get_embeddings_by_id(query_embeddings, case)
        get_similarity(query, candidate_embeddings['embeddings'])

    # todo finish


def run(dataset, model):
    eval, test = get_data_class(dataset).get_eval_data()
    query_embeddings, candidate_embeddings = load_embeddings(dataset, model)

    threshold = get_threshold(query_embeddings, candidate_embeddings, eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''case vectoriser arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    args = parser.parse_args()
    run(args.dataset, args.model_name)
