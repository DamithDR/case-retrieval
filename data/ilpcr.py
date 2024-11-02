import random

import pandas as pd
from datasets import load_dataset

from data.DataClass import DataClass


class ilpcr(DataClass):

    def __init__(self):
        self.queries = []
        self.query_ids = []
        self.candidates = []
        self.candidates_ids = []
        super().__init__('Exploration-Lab/IL-TUR')

    def load_data(self):
        self.load_candidates()
        self.load_queries()

    def load_candidates(self):
        temp_data = load_dataset(self.name, "pcr", split='test_candidates')
        case_list = temp_data['text']
        self.candidates_ids = temp_data['id']
        self.candidates = [' \n'.join(case) for case in case_list]

    def load_queries(self):
        temp_data = load_dataset(self.name, "pcr", split='test_queries')
        case_list = temp_data['text']
        self.query_ids = temp_data['id']
        self.queries = [' \n'.join(case) for case in case_list]

    def get_candidates(self):
        return self.candidates

    def get_candidate_ids(self):
        return self.candidates_ids

    def get_queries(self):
        return self.queries

    def get_query_ids(self):
        return self.query_ids

    def get_gold_data(self):
        dataset = load_dataset(self.name, "pcr", split='test_queries')
        cases = dataset['id']
        citations = dataset['relevant_candidates']

        data = dict()
        for case, citation_list in zip(cases, citations):
            if len(citation_list) > 0:
                data[case] = citation_list
        return data

    def get_training_permutations(self):
        train_candidates = load_dataset(self.name, "pcr", split='train_candidates')
        candidate_ids = train_candidates['id']
        train_queries = load_dataset(self.name, "pcr", split='train_queries')

        references = []
        cases = []
        label = []

        for id, case, relevant_candidates in zip(train_queries['id'], train_queries['text'],
                                                 train_queries['relevant_candidates']):
            text = ' \n'.join(case)

            # positive samples
            for candidate in relevant_candidates:
                if candidate != '':
                    idx = train_candidates['id'].index(candidate)
                    candidate_case = train_candidates[idx]
                    candidate_case = ' \n'.join(candidate_case['text'])
                    references.append(text)
                    cases.append(candidate_case)
                    label.append(1)

            # negative samples
            filtered_list = [elem for elem in candidate_ids if elem not in relevant_candidates]

            # Randomly select 5 elements from the filtered list
            if len(filtered_list) >= 5:
                selected_elements = random.sample(filtered_list, 5)
            else:
                selected_elements = filtered_list

            for element in selected_elements:
                idx = train_candidates['id'].index(element)
                element_text = train_candidates[idx]
                element_text = ' \n'.join(element_text['text'])
                references.append(text)
                cases.append(element_text)
                label.append(0)

        return pd.DataFrame({'reference': references, 'candidate': cases, 'label': label})

    def get_dev_permutations(self):
        dev_candidates = load_dataset(self.name, "pcr", split='dev_candidates')
        candidate_ids = dev_candidates['id']
        dev_queries = load_dataset(self.name, "pcr", split='dev_queries')

        references = []
        cases = []
        label = []

        for id, case, relevant_candidates in zip(dev_queries['id'], dev_queries['text'],
                                                 dev_queries['relevant_candidates']):
            text = ' \n'.join(case)

            # positive samples
            for candidate in relevant_candidates:
                if candidate != '':
                    idx = dev_candidates['id'].index(candidate)
                    candidate_case = dev_candidates[idx]
                    candidate_case = ' \n'.join(candidate_case['text'])
                    references.append(text)
                    cases.append(candidate_case)
                    label.append(1)

            # negative samples
            filtered_list = [elem for elem in candidate_ids if elem not in relevant_candidates]

            # Randomly select 5 elements from the filtered list
            if len(filtered_list) >= 5:
                selected_elements = random.sample(filtered_list, 5)
            else:
                selected_elements = filtered_list

            for element in selected_elements:
                idx = dev_candidates['id'].index(element)
                element_text = dev_candidates[idx]
                element_text = ' \n'.join(element_text['text'])
                references.append(text)
                cases.append(element_text)
                label.append(0)

        return pd.DataFrame({'reference': references, 'candidate': cases, 'label': label})