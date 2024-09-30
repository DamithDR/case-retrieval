import argparse
import os

import numpy as np
import pandas as pd

from data.coliee import coliee
from data.ecthr import ecthr
from data.ilpcr import ilpcr
from data.muser import muser
from models.flagembed import flag
from models.nvembedv2 import Nvembedv2
from models.sftembed import sfr
from models.stellaembed import stella


def get_data_class(dataset):
    if dataset == 'IL_PCR':
        return ilpcr()
    elif dataset == 'coliee':
        return coliee()
    elif dataset == 'ilpcr':
        return ilpcr()
    elif dataset == 'muser':
        return muser()
    elif dataset == 'ecthr':
        return ecthr()


def get_model_class(model_name):
    if model_name == 'nvidia/NV-Embed-v2':
        return Nvembedv2()
    elif model_name == 'BAAI/bge-en-icl':
        return flag()
    elif model_name == 'Salesforce/SFR-Embedding-2_R':
        return sfr()
    elif model_name == 'dunzhang/stella_en_1.5B_v5':
        return stella()


def standadise_name(name):
    return name.split('/')[-1] if name.__contains__('/') else name


def get_save_names(model_class, data_class):
    model_name = model_class.get_name()
    data_name = data_class.get_name()
    return standadise_name(model_name), standadise_name(data_name)


def save_embeddings(embeddings, ids, split_alias, model_alias, dataset_alias):
    embeddings_df = pd.DataFrame(
        {'ids': ids, 'embeddings': [np.array2string(embedding, separator=',') for embedding in embeddings]})

    save_path = f'embeddings/{model_alias}'
    if not os.path.exists(save_path): os.makedirs(save_path)

    embeddings_df.to_csv(f'{save_path}/{dataset_alias}_{split_alias}.csv', index=False)


def vectorise_candidates(model_class, data_class):
    candidate_embeddings = model_class.vectorise(data_class.get_candidates()[:10]) # todo remove after testing
    ids = data_class.get_candidate_ids()[:10]

    model_alias, dataset_alias = get_save_names(model_class, data_class)
    save_embeddings(candidate_embeddings, ids, 'candidates', model_alias, dataset_alias)


def vectorise_queries(model_class, data_class):
    query_embeddings = model_class.vectorise(data_class.get_queries()[:10])
    ids = data_class.get_query_ids()[:10]

    model_alias, dataset_alias = get_save_names(model_class, data_class)
    save_embeddings(query_embeddings, ids, 'queries', model_alias, dataset_alias)


def vectorise_dataset(model_class, data_class):
    embeddings = model_class.vectorise(data_class.get_data()[:10])
    ids = data_class.get_ids()[:10]

    model_alias, dataset_alias = get_save_names(model_class, data_class)
    save_embeddings(embeddings, ids, 'data', model_alias, dataset_alias)


def vectorise(model_class, data_class):
    if data_class.get_name() in ['data/files/irled', 'Exploration-Lab/IL-TUR']:
        vectorise_queries(model_class, data_class)
        vectorise_candidates(model_class, data_class)
    elif data_class.get_name in ['data/files/muser/muser_cases_pool.json', 'data/files/coliee',
                                 'RashidHaddad/ECTHR-PCR']:
        vectorise_dataset(model_class, data_class)


def run(args):
    print(f'running args : model - {args.model_name} | data - {args.dataset}')

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
