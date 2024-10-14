import numpy as np


def cosine_similarity(query_vector, document_vectors):
    """
    Computes the cosine similarity between a query vector and a set of document vectors.

    Args:
        query_vector (array-like): The query vector.
        document_vectors (array-like): A matrix of document vectors.

    Returns:
        np.ndarray: An array of cosine similarities between the query vector and the document vectors.
    """
    query_vector = np.asarray(query_vector).flatten()
    document_vectors = np.asarray(document_vectors)

    dot_products = document_vectors @ query_vector  # Dot product between vectors
    doc_norms = np.linalg.norm(document_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)

    denominator = np.maximum(doc_norms * query_norm,
                             1e-10)  # Avoid division by zero
    return dot_products / denominator
