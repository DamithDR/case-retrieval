import jieba

from util.name_handler import get_data_class

def count_avg_words_chinese(queries):
    total_no = len(queries)

    total_words = 0
    for query in queries:
        total_words += len(list(jieba.cut(query)))

    return round(total_words / total_no, 2)

def count_avg_words(queries):
    total_no = len(queries)

    total_words = 0
    for query in queries:
        total_words += len(query.split(' '))

    return round(total_words / total_no, 2)


def run(dataset):
    data_class = get_data_class(dataset)

    if dataset in ['irled', 'ilpcr']:
        queries = data_class.get_queries()
        candidates = data_class.get_candidates()

        print(f'dataset : {dataset}, total no of queries : {len(queries)}')
        print(f'dataset : {dataset}, total no of cadidates : {len(candidates)}')

        #
        #
        # avg_words_queries = count_avg_words(queries)
        # avg_words_candidates = count_avg_words(candidates)
        #
        # print(
        #     f'Dataset : {dataset}, avg_words_queries: {avg_words_queries}, avg_words_candidates: {avg_words_candidates}')
    elif dataset in ['muser', 'coliee', 'ecthr']:
        gold_data = data_class.get_gold_data()
        query_keys = gold_data.keys()
        candidate_lists = gold_data.values()
        candidate_keys = [candidate for candidate_list in candidate_lists for candidate in candidate_list]

        print(f'dataset : {dataset}, total no of queries : {len(query_keys)}')
        print(f'dataset : {dataset}, total no of cadidates : {len(candidate_keys)}')
        # ids = data_class.get_ids()
        # data = data_class.get_data()

        # queries = []
        # for key in query_keys:
        #     idx = ids.index(key)
        #     queries.append(data[idx])
        #
        # candidates = []
        # for key in candidate_keys:
        #     idx = ids.index(key)
        #     candidates.append(data[idx])
        # if dataset == 'muser':
        #     avg_words_queries = count_avg_words_chinese(queries)
        #     avg_words_candidates = count_avg_words_chinese(candidates)
        # else:
        #     avg_words_queries = count_avg_words(queries)
        #     avg_words_candidates = count_avg_words(candidates)
        # print(
        #     f'Dataset : {dataset}, avg_words_queries: {avg_words_queries}, avg_words_candidates: {avg_words_candidates}')


if __name__ == '__main__':
    datasets = ['irled', 'ilpcr', 'muser', 'coliee', 'ecthr']
    # datasets = ['muser']

    for dataset in datasets:
        run(dataset)
