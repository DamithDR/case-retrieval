import json
import os
import random

from data.DataClass import DataClass


class lecardv2(DataClass):

    def __init__(self):
        self.queries = []
        self.query_ids = []
        self.candidates = []
        self.candidates_ids = []
        super().__init__('data/files/lecardv2')

    def load_data(self):
        self.load_candidates()
        self.load_queries()

    def load_candidates(self):

        gold = self.get_gold_data()

        all_gold_candidates = [candidate for lst in gold.values() for candidate in lst]
        all_candidates = os.listdir(f'{self.name}/candidates/')

        sampling_candidates = set(all_candidates) - set(all_gold_candidates)
        if len(sampling_candidates) > 1000:
            selected_items = random.sample(sampling_candidates, 1000)
        else:
            selected_items = sampling_candidates
        sampled_candidates = selected_items + set(all_gold_candidates)

        for filename in sampled_candidates:
            if filename.endswith('.json'):
                file_path = os.path.join(f'{self.name}/candidates/', filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    text = data['qw']
                self.candidates_ids.append(filename)
                self.candidates.append(text)

    def load_queries(self):
        with open('data/files/lecardv2/queries/test_query.json', 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                self.query_ids.append(data['id'])
                self.queries.append(data['query'])

    def get_candidates(self):
        return self.candidates

    def get_candidate_ids(self):
        return self.candidates_ids

    def get_queries(self):
        return self.queries

    def get_query_ids(self):
        return self.query_ids

    def get_gold_data(self):
        with open('data/gold_annot/lecardv2.trec', 'r') as file:
            data = dict()
            for line in file:
                annot = line.split('\t')
                ref = annot[0]
                relevant = f'{annot[2]}.json'
                if not data.keys().__contains__(ref):
                    data[ref] = []
                data[ref].append(relevant)
        return data
