from abc import abstractmethod


class DataClass:
    def __init__(self):
        self.name = None
        self.queries = None
        self.query_ids = None
        self.candidates = None
        self.candidates_ids = None

        self.load_candidates()
        self.load_queries()

    @abstractmethod
    def load_candidates(self):
        pass

    @abstractmethod
    def load_queries(self):
        pass

    def get_candidates(self):
        return self.candidates

    def get_candidate_ids(self):
        return self.candidates_ids

    def get_queries(self):
        return self.queries

    def get_query_ids(self):
        return self.query_ids

    def get_name(self):
        return self.name
