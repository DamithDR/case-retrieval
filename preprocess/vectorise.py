import argparse

from data.IL_PCR import IL_PCR


def get_data_class(dataset, max_length):
    if dataset == 'IL_PCR':
        return IL_PCR(max_length)


def run(args):
    print(args)

    data_class = get_data_class(args.dataset, args.max_length)
    data_class.vectorise_candidates(args.model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''case retrieval arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--max_length', type=int, default=32768, required=False, help='max_length')
    args = parser.parse_args()
    run(args)
