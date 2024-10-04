from numpy import dot
from numpy.linalg import norm


def cosine_similarity(vec1, vec2):
    cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return cos_sim


def sort_by_numbers_desc(numbers, labels):
    sorted_pairs = sorted(zip(numbers, labels), key=lambda x: x[0], reverse=True)
    sorted_numbers, sorted_labels = zip(*sorted_pairs)
    return list(sorted_numbers), list(sorted_labels)
