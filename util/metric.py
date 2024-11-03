def recall_at_k(recommended_items, relevant_items, k):
    """
    Calculate Recall@K

    Args:
        recommended_items (list): List of items recommended by the model.
        relevant_items (list): List of relevant (ground-truth) items.
        k (int): Number of top recommendations to consider.

    Returns:
        float: Recall@K value.
    """
    # Get top-k recommended items
    top_k_recommendations = recommended_items[:k]

    # Count the relevant items in the top-k recommendations
    relevant_in_top_k = len(set(top_k_recommendations) & set(relevant_items))

    # Calculate recall@k
    recall = relevant_in_top_k / len(relevant_items) if relevant_items else 0

    return recall


def precision_at_k(recommended_items, relevant_items, k):
    """
    Calculate Precision@K

    Args:
        recommended_items (list): List of items recommended by the model.
        relevant_items (list): List of relevant (ground-truth) items.
        k (int): Number of top recommendations to consider.

    Returns:
        float: Precision@K score.
    """
    # Get top-k recommended items
    top_k_recommendations = recommended_items[:k]

    # Count the number of relevant items in the top-k recommendations
    relevant_in_top_k = len(set(top_k_recommendations) & set(relevant_items))

    # Calculate precision@k
    precision = relevant_in_top_k / k if k > 0 else 0

    return precision


def f1_at_k(recommended_items, relevant_items, k):
    """
    Calculate F1@K

    Args:
        recommended_items (list): List of items recommended by the model.
        relevant_items (list): List of relevant (ground-truth) items.
        k (int): Number of top recommendations to consider.

    Returns:
        float: F1@K score.
    """
    # Get top-k recommended items
    top_k_recommendations = recommended_items[:k]

    # Count the number of relevant items in the top-k recommendations
    relevant_in_top_k = len(set(top_k_recommendations) & set(relevant_items))

    # Calculate precision@k
    precision = relevant_in_top_k / k if k > 0 else 0

    # Calculate recall@k
    recall = relevant_in_top_k / len(relevant_items) if relevant_items else 0

    # Calculate F1@K (harmonic mean of precision and recall)
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def average_precision(recommended_items, relevant_items):
    """
    Calculate the Average Precision (AP) for a single queries.

    Args:
        recommended_items (list): List of items recommended by the model.
        relevant_items (list): List of relevant (ground-truth) items.

    Returns:
        float: Average Precision for the given recommendation.
    """
    hits = 0  # Number of relevant items encountered
    sum_precisions = 0  # Sum of precision values at relevant positions

    for idx, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            hits += 1
            precision_at_k = hits / idx  # Precision at this rank
            sum_precisions += precision_at_k

    if hits == 0:
        return 0  # If there are no relevant items, return 0

    return sum_precisions / len(relevant_items)  # Average precision


def mean_average_precision(recommended_lists, relevant_lists):
    """
    Calculate the Mean Average Precision (MAP) for multiple queries.

    Args:
        recommended_lists (list of lists): List of recommended items for each queries.
        relevant_lists (list of lists): List of relevant (ground-truth) items for each queries.

    Returns:
        float: Mean Average Precision (MAP) score.
    """
    ap_scores = []  # List to store AP scores for each queries

    for recommended_items, relevant_items in zip(recommended_lists, relevant_lists):
        ap = average_precision(recommended_items, relevant_items)
        ap_scores.append(ap)

    return sum(ap_scores) / len(ap_scores) if ap_scores else 0  # Return the mean AP score

