import argparse
import os

import numpy as np
import pandas as pd

from data.IL_PCR import IL_PCR
from models.nvembedv2 import Nvembedv2


def get_data_class(dataset):
    if dataset == 'IL_PCR':
        return IL_PCR()


def get_model_class(model_name):
    if model_name == 'nvidia/NV-Embed-v2':
        return Nvembedv2()


def standadise_name(name):
    return name.split('/')[-1] if name.__contains__('/') else name


def get_save_names(model_class, data_class):
    model_name = model_class.get_name()
    data_name = data_class.get_name()
    return standadise_name(model_name), standadise_name(data_name)


def save_embeddings(embeddings, ids, split_alias, model_alias, dataset_alias):

    print(f'ids {len(ids)} | embeddings {len(embeddings)}')
    print(embeddings)
    # print(np.array2string(embeddings, separator=','))
    embeddings_df = pd.DataFrame(
        {'ids': ids, 'embeddings': [np.array2string(embedding, separator=',') for embedding in embeddings]})

    save_path = f'embeddings/{model_alias}/{dataset_alias}_{split_alias}.csv'
    os.makedirs(save_path)
    embeddings_df.to_csv(save_path, index=False)


def vectorise_candidates(model_class, data_class):
    candidate_embeddings = model_class.vectorise(data_class.get_candidates()[:10]) #todo remove after testing
    ids = data_class.get_candidate_ids()[:10]

    model_alias, dataset_alias = get_save_names(model_class, data_class)
    save_embeddings(candidate_embeddings, ids, 'candidates', model_alias, dataset_alias)


def vectorise_queries(model_class, data_class):
    query_embeddings = model_class.vectorise(data_class.get_queries()[:10])
    ids = data_class.get_query_ids()[:10]

    model_alias, dataset_alias = get_save_names(model_class, data_class)
    save_embeddings(query_embeddings, ids, 'queries', model_alias, dataset_alias)


def run(args):
    print(args)

    data_class = get_data_class(args.dataset)
    model_class = get_model_class(args.model_name)

    vectorise_queries(model_class, data_class)
    vectorise_candidates(model_class, data_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''case vectoriser arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    args = parser.parse_args()
    run(args)
