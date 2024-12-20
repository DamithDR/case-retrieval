import argparse
import os

import numpy as np

from util.name_handler import get_data_class, get_model_class, get_save_names


def save_embeddings(embeddings, ids, model_alias, dataset_alias):
    save_path = f'embeddings/{model_alias}/{dataset_alias}'
    if not os.path.exists(save_path): os.makedirs(save_path)

    for id, embedding in zip(ids, embeddings):
        id = str(id).replace('/', '-')
        np.save(f'{save_path}/{id}.npy', embedding)

    print(f'saving complete : {save_path}/')


def vectorise_candidates(model_class, data_class):
    candidate_embeddings = model_class.vectorise(data_class.get_candidates())
    ids = data_class.get_candidate_ids()

    model_alias, dataset_alias = get_save_names(model_class, data_class)
    save_embeddings(candidate_embeddings, ids, model_alias, dataset_alias)


def vectorise_queries(model_class, data_class):
    query_embeddings = model_class.vectorise(data_class.get_queries())
    ids = data_class.get_query_ids()

    model_alias, dataset_alias = get_save_names(model_class, data_class)
    save_embeddings(query_embeddings, ids, model_alias, dataset_alias)


def vectorise_dataset(model_class, data_class):
    embeddings = model_class.vectorise(data_class.get_data())
    ids = data_class.get_ids()

    model_alias, dataset_alias = get_save_names(model_class, data_class)
    save_embeddings(embeddings, ids, model_alias, dataset_alias)


def vectorise(model_class, data_class, dataset):
    if dataset in ['irled', 'ilpcr','lecardv2']:
        vectorise_queries(model_class, data_class)
        vectorise_candidates(model_class, data_class)
    elif dataset in ['muser', 'coliee', 'ecthr']:
        vectorise_dataset(model_class, data_class)


def run(args):
    print(f'running args : model - {args.model_name} | data - {args.dataset}')

    data_class = get_data_class(args.dataset)
    model_class = get_model_class(args.model_name)

    vectorise(model_class, data_class, args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''case vectoriser arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    args = parser.parse_args()
    run(args)
