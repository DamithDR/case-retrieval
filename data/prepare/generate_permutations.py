import os.path

from util.name_handler import get_data_class


def run(dataset):
    if not os.path.exists(f'data/prepare/{dataset}.csv'):
        data_class = get_data_class(dataset)
        df = data_class.get_training_permutations()
        df.to_csv(f'{dataset}.csv', index=False)
    if not os.path.exists(f'data/prepare/{dataset}_dev.csv'):
        data_class = get_data_class(dataset)
        df = data_class.get_dev_permutations()
        df.to_csv(f'{dataset}_dev.csv', index=False)


if __name__ == '__main__':
    run('ilpcr')
