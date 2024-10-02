from numpy import dot
from numpy.linalg import norm


def cosine_similarity(vec1, vec2):
    cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return cos_sim
