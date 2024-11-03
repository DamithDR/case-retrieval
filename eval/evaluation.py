import argparse
import os

import numpy as np

from util.metric import recall_at_k, mean_average_precision, f1_at_k, precision_at_k
from util.name_handler import get_data_class, standadise_name, get_embedding_folder
from util.similarity import cosine_similarity, sort_by_numbers_desc

numbers = [1] + list(range(5, 51, 5)) + [100]


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
    case = case.replace('/', '-')
    q_embed = query_embeddings[case]
    for key, embedding in candidate_embeddings.items():
        if key != case:
            similarity = cosine_similarity(q_embed, embedding)
            similarity_scores.append(similarity)
            candidate_keys.append(key)
    return candidate_keys, similarity_scores


def calculate_metrics(query_embeddings, candidate_embeddings, test):
    results_dict = {}
    for case, citations in test.items():
        case = case.replace('/', '-')
        keys, similarity = get_similarity(case, query_embeddings, candidate_embeddings)
        results_dict[case] = {'keys': keys, 'similarity': similarity}

    gold = []
    predictions = []

    f1_values = {}
    p_values = {}
    r_values = {}

    print(f"eval k values = {numbers}")

    for case, citations in test.items():
        case = case.replace('/', '-')  # for ecthr dataset
        citations = [citation.replace('/', '-') for citation in citations]  # for ecthr dataset
        gold.append(citations)
        results = results_dict[case]
        values, labels = sort_by_numbers_desc(results['similarity'], results['keys'])
        predictions.append(labels)

        for number in numbers:
            if number not in f1_values.keys():
                f1_values[number] = []
                p_values[number] = []
                r_values[number] = []
            f1_values[number].append(f1_at_k(labels, citations, number))
            p_values[number].append(precision_at_k(labels, citations, number))
            r_values[number].append(recall_at_k(labels, citations, number))

    MAP = mean_average_precision(predictions, gold)
    f1_final = []
    p_final = []
    r_final = []
    for number in numbers:
        f1_final.append(np.mean(f1_values[number]).item())
        p_final.append(np.mean(p_values[number]).item())
        r_final.append(np.mean(r_values[number]).item())

    return MAP, f1_final, p_final, r_final


def run(dataset, model):
    data_class = get_data_class(dataset)
    gold = data_class.get_gold_data()
    candidate_embeddings = load_embeddings(dataset, model)

    print(len(candidate_embeddings))

    print(candidate_embeddings.keys())
    query_embeddings = dict()
    if dataset == 'ilpcr' or dataset == 'irled' or dataset == 'lecardv2':
        query_ids = data_class.get_query_ids()
        query_embeddings = {key: candidate_embeddings.pop(key) for key in query_ids}
    elif dataset == 'coliee' or dataset == 'muser' or dataset == 'ecthr':
        query_embeddings = candidate_embeddings
    MAP, f1_final, p_final, r_final = calculate_metrics(query_embeddings, candidate_embeddings, gold)
    MAP = round(MAP, 2)
    f1_final = [str(round(f1, 2)) for f1 in f1_final]
    p_final = [str(round(p, 2)) for p in p_final]
    r_final = [str(round(r, 2)) for r in r_final]

    results_file_name = 'new_results.csv'
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
        f.write(f'{model},{dataset},F1,{MAP},{f1_results}\n')
        f.write(f'{model},{dataset},Precision,{MAP},{p_results}\n')
        f.write(f'{model},{dataset},Recall,{MAP},{r_results}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluator arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    args = parser.parse_args()

    run(args.dataset, args.model_name)

    # datasets = ['ilpcr', 'coliee', 'irled', 'muser','ecthr']
    # models = ['legalbertfinetuned']
    # models = ['BAAI/bge-en-icl', 'Salesforce/SFR-Embedding-2_R', 'dunzhang/stella_en_1.5B_v5','nlpaueb/legal-bert-base-uncased']
    #
    # for model in models:
    #     for dataset in datasets:
    #         run(dataset, model)
