from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree,LSHForest
from scipy.spatial import KDTree
import numpy as np
import hnswlib


def brute_force_search(query_vector, dataset_vectors, top_k=5):
    similarities = cosine_similarity(query_vector.reshape(1, -1), dataset_vectors)
    indices = similarities.argsort()[0][-top_k:][::-1]
    return indices, similarities[0][indices]

def ball_tree_search(query_vector, dataset_vectors, top_k=5):
    tree = BallTree(dataset_vectors)
    distances, indices = tree.query(query_vector.reshape(1, -1), k=top_k)
    return indices[0], distances[0]


def kd_tree_search(query_vector, dataset_vectors, top_k=5):
    tree = KDTree(dataset_vectors)
    distances, indices = tree.query(query_vector.reshape(1, -1), k=top_k)
    return indices[0], distances[0]

def lsh_search(query_vector, dataset_vectors, top_k=5):
    lshf = LSHForest(n_estimators=10)
    lshf.fit(dataset_vectors)
    distances, indices = lshf.kneighbors(query_vector.reshape(1, -1), n_neighbors=top_k)
    return indices[0], distances[0]

def product_quantization_search(query_vector, dataset_vectors, top_k=5, num_clusters=4):
    num_subvectors = 4  # Example: splitting the vector into 4 subvectors
    subvector_dim = len(query_vector) // num_subvectors
    subvectors = [query_vector[i:i+subvector_dim] for i in range(0, len(query_vector), subvector_dim)]

    codebook = []
    for subvector in subvectors:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters)
        kmeans.fit(dataset_vectors)
        codebook.append(kmeans.cluster_centers_)

    quantized_query = []
    for subvector, subvector_codebook in zip(subvectors, codebook):
        distances = euclidean_distances([subvector], subvector_codebook)
        quantized_query.append(subvector_codebook[np.argmin(distances)])

    return brute_force_search(np.concatenate(quantized_query), dataset_vectors, top_k)

def hnsw_search(query_vector, dataset_vectors, top_k=5, ef=50):
    dim = len(query_vector)
    num_elements = len(dataset_vectors)

    p = hnswlib.Index(space='l2', dim=dim)  # L2 space for Euclidean distance
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    p.set_ef(ef)

    for i, vector in enumerate(dataset_vectors):
        p.add_items(np.array([vector]))

    p.set_ef(ef)
    labels, distances = p.knn_query(np.array([query_vector]), k=top_k)
    return labels[0], distances[0]