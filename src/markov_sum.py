from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

def create_transition_matrix(
    processed_data: List[Dict[str, Any]], 
    n_clusters: int , 
) -> np.ndarray:
    """
    Creates a transition matrix based on the transitions between clusters.

    Args:
        processed_data (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
            - "cluster" (int): The cluster label assigned by k-means.
            - "chunk" (str): The chunk itself.
            - "embedding" (np.ndarray): The embedding of the chunk.
            - "pos" (int): The chronological position of the chunk.
        n_clusters (int): The number of clusters.

    Returns:
        np.ndarray: A transition matrix where each entry (i, j) represents the transition probability from cluster i to cluster j.
    """
    # Initialize the transition matrix with zeros
    transition_matrix = np.zeros(shape=(n_clusters, n_clusters))

    # Iterate over the processed_data to fill the transition matrix
    for i in tqdm(range(len(processed_data) - 1), desc="Calculating transition matrix..."):
        from_cluster = processed_data[i]["cluster"]
        to_cluster = processed_data[i + 1]["cluster"]
        transition_matrix[from_cluster][to_cluster] += 1

    # Calculate the sum of transitions from each cluster (row-wise sum)
    row_sums = np.sum(transition_matrix, axis=1, keepdims=True)

    # Normalize the transition matrix by row sums to get the probabilities
    transition_matrix = transition_matrix / row_sums

    return transition_matrix




def plot_networkx_graph(G: nx.Graph, title: str = "NetworkX Graph", node_size: int = 300, font_size: int = 10) -> None:
    """
    Plots a NetworkX graph using Matplotlib.

    Args:
        G (nx.Graph): The NetworkX graph to plot.
        title (str, optional): Title of the graph plot. Defaults to "NetworkX Graph".
        node_size (int, optional): Size of the nodes in the plot. Defaults to 300.
        font_size (int, optional): Font size of the node labels. Defaults to 10.

    Returns:
        None
    """
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)  # Positions the nodes using the spring layout algorithm

    # Draw nodes, edges, and labels
    nx.draw(G, pos, with_labels=True, node_size=node_size, font_size=font_size, node_color='skyblue', edge_color='gray', font_weight='bold')

    plt.title(title)
    plt.show()



def create_directed_graph(
    transition_matrix: np.ndarray, 
    summary_by_cluster: Dict[int, str],
    plot : bool = True 
) -> nx.DiGraph:
    """
    Creates a directed graph from a transition matrix and summaries by cluster.

    Args:
        transition_matrix (np.ndarray): A transition matrix where each entry (i, j) represents the transition probability from cluster i to cluster j.
        summary_by_cluster (Dict[int, str]): A dictionary where keys are cluster indices, and values are summaries corresponding to those clusters.

    Returns:
        nx.DiGraph: A directed graph with edges weighted by transition probabilities and nodes labeled by summaries.
    """
    # Initialize a directed graph
    G = nx.DiGraph()

    # Create a lookup to convert cluster index to its corresponding summary
    cluster_by_intent = {intent: int(cluster) for cluster, intent in summary_by_cluster.items()}

    # Add edges with weights
    for i, from_intent in summary_by_cluster.items():
        for j, to_intent in summary_by_cluster.items():
            G.add_edge(from_intent, to_intent, weight=transition_matrix[int(i), int(j)])
    
    if plot : 
        plot_networkx_graph(G)
    return G 


def find_most_probable_sequence(T):
    """
    Finds the most probable Hamiltonian path in a Markov chain given the transition matrix T.
    
    Args:
    - T (list of list of floats): The transition matrix where T[i][j] represents the probability of transitioning from node i to node j.
    
    Returns:
    - max_prob (float): The maximum probability of the Hamiltonian path.
    - path (list of int): The most probable path visiting all nodes.
    """
    n = len(T)
    dp = [[0 for _ in range(n)] for _ in range(2**n)]  # DP table to store maximum probabilities
    parent = [[-1 for _ in range(n)] for _ in range(2**n)]  # Table to reconstruct the path

    # Initialize base cases: starting from each node with probability 1
    for i in range(n):
        dp[1 << i][i] = 1  # Start at node i with probability 1

    # Fill the DP table
    for S in range(1, 2**n):  # Iterate over all subsets of nodes
        for i in range(n):  # Consider each possible ending node i
            if not (S & (1 << i)):  # If i is not in subset S, skip
                continue
            for j in range(n):  # Consider transitions to node i
                if i == j or not (S & (1 << j)):  # Skip if i == j or j is not in S
                    continue
                # Calculate maximum probability path ending at i
                prob = dp[S - (1 << i)][j] * T[j][i]
                if prob > dp[S][i]:
                    dp[S][i] = prob
                    parent[S][i] = j  # Keep track of the path

    # Determine the maximum probability path
    max_prob = 0
    end_node = -1
    for i in range(n):
        if dp[(1 << n) - 1][i] > max_prob:
            max_prob = dp[(1 << n) - 1][i]
            end_node = i

    # Reconstruct the most probable path
    path = []
    current_set = (1 << n) - 1
    current_node = end_node

    while current_node != -1:
        path.append(current_node)
        next_node = parent[current_set][current_node]
        current_set = current_set - (1 << current_node)
        current_node = next_node

    path.reverse()  # Reverse to get the path from start to end

    return path ,max_prob





