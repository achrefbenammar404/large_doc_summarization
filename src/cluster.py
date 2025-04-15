from typing import List, Tuple, Dict, Any , Union
import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI
import json
from src.logger.logger import get_logger 
from typing import Dict, List
from src.llm.llm import CollectionLLM
from sklearn.metrics import silhouette_score

logger = get_logger(__file__)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Any, Tuple

def cluster_chunks_kmeans(
    vdb: List[Dict[str, Any]], 
    cluster_range: Tuple[int, int] = (5, 10)
) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
    """
    Finds the best number of clusters using silhouette score and clusters the embeddings using k-means++.

    Args:
        vdb (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
            - "pos" (int): The chronological position of the chunk.
            - "chunk" (str): The chunk itself.
            - "embedding" (np.ndarray): The embedding of the chunk.
        cluster_range (Tuple[int, int], optional): The range of cluster numbers to evaluate. Defaults to (5, 10).

    Returns:
        Tuple[List[Dict[str, Any]], np.ndarray, int]: 
            - List[Dict[str, Any]]: A list of dictionaries where each dictionary contains:
                - "cluster" (int): The cluster label assigned by k-means.
                - "chunk" (str): The chunk itself.
                - "embedding" (np.ndarray): The embedding of the chunk.
                - "pos" (int): The chronological position of the chunk.
            - np.ndarray: An array of cluster centroids.
            - int: The optimal number of clusters.
    """
    # Extract embeddings from the provided dictionary list
    embeddings = np.array([d["embedding"] for d in vdb])
    n_samples = embeddings.shape[0]

    if n_samples < 2:
        raise ValueError(f"Number of samples ({n_samples}) is too small for clustering.")

    # Adjust cluster_range based on n_samples
    min_clusters = max(2, cluster_range[0])
    max_clusters = min(cluster_range[1], n_samples - 1)

    if min_clusters > max_clusters:
        min_clusters = max(2, min_clusters)
        max_clusters = min_clusters  # Set max_clusters to min_clusters to have only one possible cluster

    print(f"Clustering with number of clusters ranging from {min_clusters} to {max_clusters} based on n_samples={n_samples}.")

    best_num_clusters = min_clusters
    best_score = -1
    best_kmeans = None

    # Find the best number of clusters using silhouette score
    for num_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
        kmeans.fit(embeddings)
        cluster_labels = kmeans.labels_
        
        try:
            score = silhouette_score(embeddings, cluster_labels)
            print(f"Number of clusters: {num_clusters}, Silhouette Score: {score:.4f}")
        except ValueError as ve:
            print(f"Skipping {num_clusters} clusters due to error: {ve}")
            continue

        if score > best_score:
            best_score = score
            best_num_clusters = num_clusters
            best_kmeans = kmeans

    if best_kmeans is None:
        raise ValueError("Failed to find a suitable number of clusters.")

    # Using the best number of clusters to fit the final k-means model
    best_kmeans.fit(embeddings)
    cluster_labels = best_kmeans.labels_
    cluster_centroids = best_kmeans.cluster_centers_

    # Create a list of dictionaries with cluster information
    clustered_chunks = [
        {
            "cluster": int(cluster_labels[i]),
            "chunk": data_point["chunk"],
            "embedding": data_point["embedding"],
            "pos": data_point["pos"]
        }
        for i, data_point in enumerate(vdb)
    ]

    print(f"Optimal number of clusters: {best_num_clusters} with Silhouette Score: {best_score:.4f}")

    return clustered_chunks, cluster_centroids, best_num_clusters



def find_closest_data_points_to_centroid(
    clustered_chunks: List[Dict[str, Any]], 
    cluster_centroids: np.ndarray, 
    top_k: int
) -> Dict[int, List[str]]:
    """
    Finds the closest data points to each cluster centroid.

    Args:
        clustered_chunks (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
            - "cluster" (int): The cluster label assigned by k-means.
            - "chunk" (str): The chunk itself.
            - "embedding" (np.ndarray): The embedding of the chunk.
            - "pos" (int): The chronological position of the chunk.
        cluster_centroids (np.ndarray): An array of cluster centroids from k-means.
        top_k (int): The number of closest data points to return for each cluster.

    Returns:
        Dict[int, List[str]]: A dictionary where the keys are cluster indices, and the values are lists 
                               of the closest `top_k` chunks to the respective cluster centroids.
    """
    closest_sentences = {}

    for cluster_index, cluster_centroid in enumerate(cluster_centroids):
        distances = []

        for data in clustered_chunks:
            if data["cluster"] == cluster_index:
                distance = np.linalg.norm(cluster_centroid - data["embedding"])
                distances.append({
                    "distance": distance,
                    "chunk": data["chunk"]
                })

        # Sort distances in ascending order
        distances = sorted(distances, key=lambda x: x["distance"])
        # Store the top_k closest chunks for the current cluster
        closest_sentences[cluster_index] = [e["chunk"] for e in distances[:top_k]]

    return closest_sentences



def aggregate_summaries_for_each_cluster(
    closest_chunks: Dict[int, List[str]], 
    sys_prompt_content: str , 
    client : OpenAI, 
    model : str 
) -> Dict[int, str]:
    """
    Aggregates summaries for each cluster based on the closest chunks and system prompt content.

    Args:
        closest_chunks (Dict[int, List[str]]): A dictionary where each key is a cluster index and the value is a list of sentences (chunks) closest to the cluster centroid.
        sys_prompt_content (str): The content of the system prompt to be used for summarization.

    Returns:
        Dict[int, str]: A dictionary where each key is a cluster index and the value is the summarized text for that cluster.
    """
    try : 
        llm = CollectionLLM.llm_collection[model]
    except Exception as e : 
        logger.error(f"model {model} is not included into our llm collection. {e}")
    aggregated_messages = [{"role": "system", "content": sys_prompt_content}]
    cluster_summaries = {}

    for cluster_index, sentences in closest_chunks.items():
        chunks = " \n - ".join(sentences)
        format = "{ 'summary': 'your output goes directly here' }"
        prompt = (
            f"You are an AI assistant designed to summarize text. You will be provided with multiple sentences extracted "
            f"from a large document that may be incomplete or incoherent at first sight. You need to focus and try to extract "
            f"the hidden main ideas. Your task is to summarize these sentences into a coherent single paragraph that captures "
            f"the main ideas. The sentences are:\n - {chunks}\nYou answer in this format {format}, you only output the needed format no more no less please."
        )
        messages = aggregated_messages + [{"role": "user", "content": prompt}]
        summary = llm.get_response(messages=messages)

        summary_text = str(summary)

        cluster_summaries[cluster_index] = summary_text
        aggregated_messages.append({"role": "assistant", "content": summary_text})

    return cluster_summaries

import json
import logging
import re
from typing import Dict, Union

logger = logging.getLogger(__name__)

def preprocess_llm_response(llm_message: str) -> Dict[str, Union[str, Dict]]:
    """
    Preprocess the function calling message to remove any unwanted strings,
    ensuring only the JSON part remains for parsing.

    Args:
        llm_message (str): The raw message containing the function call.

    Returns:
        Dict[str, Union[str, Dict]]: The cleaned dictionary representing the function call, or the original message if parsing fails.
    """
    # Finding the boundaries of the JSON string
    start_idx = llm_message.find('{')
    end_idx = llm_message.rfind('}')
    
    # Replace single quotes with double quotes to ensure valid JSON
    llm_message = llm_message.replace("'", '"')
    
    try:
        # Ensure we have a valid substring to parse
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = llm_message[start_idx:end_idx + 1]
            
            # Use a regular expression to clean and validate JSON
            json_str = re.sub(r'(?<!\\)\'', '"', json_str)  # Replace unescaped single quotes with double quotes
            json_str = re.sub(r'\s+', ' ', json_str).strip()  # Remove extra whitespace

            # Attempt to parse the JSON
            func_call = json.loads(json_str)
            return func_call
        else:
            logger.error("Invalid JSON structure: missing curly braces or improperly formatted.")
            print(f"Invalid JSON format in llm response: {llm_message}")
            return {'error': 'Invalid JSON structure'}
    
    except json.JSONDecodeError as e:
        # Log the error with specific details
        logger.error(f"JSON decoding error occurred in preprocess_llm_response: {e}")
        logger.error(f"Failed JSON string: {json_str}")
        print(f"JSON decoding error in llm response: {llm_message}")
        return {'error': 'JSON decoding error', 'message': llm_message}

    except Exception as e:
        # Log the general error with details
        logger.error(f"Unexpected error occurred in preprocess_llm_response: {e}")
        print(f"Unexpected error with llm response: {llm_message}")
        return {'error': 'Unexpected error', 'message': llm_message}

