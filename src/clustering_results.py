import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from typing import Dict, List, Any, Union
from src.logger.logger import get_logger
import pandas as pd
from src.indexing import *
from src.cluster import *
from src.markov_sum import *
from src.evaluation.metrics import *

from openai import OpenAI
from typing import Dict, Any
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
from tqdm import tqdm

logger = get_logger(__file__)





CHUNKING_METHODS = ["recursive"]
K_RANGE = (5 , 20)


def clustering_benchmark_pipeline(
    chunking_params: Dict[str, Any],
    embed_model_name: str,
    doc: str, 
) -> Dict[str, Any]:
    
    words = doc.split()
    chunks, avg_chunk_length = iterative_chunk(words, **chunking_params)
    logger.info(f"Document chunked using recursive method: {len(chunks)} chunks created")
   
    # Create vector database
    vector_database = create_vbd(chunks, emb_model=embed_model_name)
    logger.info("Vector database created")

    # Extract embeddings
    embeddings = np.array([dp['embedding'] for dp in vector_database])

    # Initialize results dict
    results = {}

    # K-Means++ clustering with max silhouette score
    best_k, best_score, best_labels = find_best_kmeans_k(embeddings, k_range=K_RANGE)
    cluster_labels = best_labels
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    num_clusters = len(unique_labels)
    cluster_sizes = counts
    mean_cluster_size = np.mean(cluster_sizes)
    std_cluster_size = np.std(cluster_sizes)

    results['kmeans'] = {
        'num_clusters': num_clusters,
        'cluster_sizes': cluster_sizes.tolist(),
        'mean_cluster_size': mean_cluster_size,
        'std_cluster_size': std_cluster_size,
        'silhouette_score': best_score,
        'best_k': best_k
    }

    # Agglomerative clustering with max silhouette score
    best_k, best_score, best_labels = find_best_agglomerative_k(embeddings, k_range=K_RANGE)
    cluster_labels = best_labels
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    num_clusters = len(unique_labels)
    cluster_sizes = counts
    mean_cluster_size = np.mean(cluster_sizes)
    std_cluster_size = np.std(cluster_sizes)

    results['agglomerative'] = {
        'num_clusters': num_clusters,
        'cluster_sizes': cluster_sizes.tolist(),
        'mean_cluster_size': mean_cluster_size,
        'std_cluster_size': std_cluster_size,
        'silhouette_score': best_score,
        'best_k': best_k
    }

    # DBSCAN clustering
    eps_range = np.linspace(0.01, 1, 50)  # Adjust the range as needed
    min_samples_range = range(1 , 5)
    best_eps, best_min_samples, best_score, best_labels = find_best_dbscan(embeddings, eps_range, min_samples_range)
    cluster_labels = best_labels
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Exclude noise points (-1 label)
    cluster_sizes = counts
    mean_cluster_size = np.mean(cluster_sizes)
    std_cluster_size = np.std(cluster_sizes)

    results['dbscan'] = {
        'num_clusters': num_clusters,
        'cluster_sizes': cluster_sizes.tolist(),
        'mean_cluster_size': mean_cluster_size,
        'std_cluster_size': std_cluster_size,
        'silhouette_score': best_score,
        'best_eps': best_eps,
        'best_min_samples': best_min_samples
    }

    return results

def find_best_kmeans_k(embeddings, k_range=(2, 10)):
    n_samples = embeddings.shape[0]
    max_k = min(k_range[1], n_samples - 1)  # Ensure max_k is less than n_samples
    best_k = None
    best_score = -1
    best_labels = None
    for k in range(k_range[0], max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(embeddings)
        # Check if the number of labels is valid
        n_labels = len(np.unique(labels))
        if n_labels >= 2 and n_labels <= n_samples - 1:
            try:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
            except ValueError as e:
                # Handle exceptions related to invalid silhouette score computation
                pass
    if best_labels is None:
        # Assign all samples to one cluster if no valid clustering is found
        best_labels = np.zeros(n_samples, dtype=int)
        best_k = 1
        best_score = -1
    return best_k, best_score, best_labels


def find_best_agglomerative_k(embeddings, k_range=(2, 10)):
    n_samples = embeddings.shape[0]
    max_k = min(k_range[1], n_samples - 1)
    best_k = None
    best_score = -1
    best_labels = None
    for k in range(k_range[0], max_k + 1):
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(embeddings)
        n_labels = len(np.unique(labels))
        if n_labels >= 2 and n_labels <= n_samples - 1:
            try:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
            except ValueError as e:
                pass
    if best_labels is None:
        best_labels = np.zeros(n_samples, dtype=int)
        best_k = 1
        best_score = -1
    return best_k, best_score, best_labels


def find_best_dbscan(embeddings, eps_range, min_samples_range):
    best_score = -1
    best_eps = None
    best_min_samples = None
    best_labels = None
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(embeddings)
            # Exclude cases where all points are assigned to one cluster
            if len(set(labels)) > 1:
                try:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
                        best_labels = labels
                except ValueError as e:
                    # Handle cases where silhouette_score cannot be computed
                    pass
    if best_labels is None:
        # No valid clustering found; assign all points to a single cluster
        best_labels = np.zeros(len(embeddings), dtype=int)
        best_score = -1
        best_eps = None
        best_min_samples = None
    return best_eps, best_min_samples, best_score, best_labels


if __name__=="__main__" : 

    from src.data.hf_cleaned_booksum import DataLoader
    data_set = 'test'
    # Initialize data loader
    data_loader = DataLoader(data_set=data_set)
    number_of_books = data_loader.get_number_of_books()
    logger.debug(f"Number of Books to summarize is {number_of_books}")
    chunking_params = {
        'max_length': 500,  # example value
        'overlap': 20     # example value
    }
    embed_model_name = "nomic-embed-text"
    clustering_output_dir = os.path.join('output' , "clustering_benchmark" , f'pipeline_results_{data_set}')
    os.makedirs(clustering_output_dir, exist_ok=True)
    clustering_benchmark_results = []
    for index_to_fetch in range(number_of_books):
    
        
        book_summary_couple = data_loader.get_book_summary_couple(index_to_fetch)
        doc = book_summary_couple["book"]
        reference_summary = book_summary_couple["summary"]
        
        result = clustering_benchmark_pipeline(chunking_params=chunking_params , embed_model_name=embed_model_name , doc = doc )
        clustering_benchmark_results.append(result)
        
        json_filename = os.path.join(clustering_output_dir, f'cluster_{index_to_fetch}.json')
        with open(json_filename, 'w') as json_file:
            json.dump(result, json_file, indent=4)

    df = pd.DataFrame(clustering_benchmark_results)
    df.to_csv(os.path.join(clustering_output_dir , f'our_approach_summarization_results_{data_set}.csv'), index=False)
