from typing import Dict, List, Any, Union
from src.logger.logger import get_logger
import pandas as pd
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import random
from src.indexing import *
from src.cluster import *
from src.markov_sum import *
from src.evaluation.metrics import *
from openai import OpenAI

logger = get_logger(__file__)





CHUNKING_METHODS = ["recursive", "semantic"]
def pipeline(
    chunking_method: str,
    chunking_params: Dict[str, Any],
    embed_model_name: str,
    summ_model_name: str,
    doc: str,
    top_k: int,  # top k closest to cluster centroid
    system_prompt_aggregate_summaries: str,
    system_prompt_docsummary: str,
    llm_instructions_doc_summary: str,
    reference_summary: str,
    client : OpenAI  ,  
    log: bool = False,
) -> Dict[str, Any]:
    """
    A summarization pipeline that processes a document through chunking, embedding, clustering,
    summarizing, and evaluating to generate a comprehensive summary.

    Args:
        chunking_method (str): The method used to chunk the document ('recursive' or 'semantic').
        chunking_params (Dict[str, Any]): Parameters for the chosen chunking method.
            If 'recursive', requires 'max_length' and 'overlap'.
            If 'semantic', requires 'threshold' and 'model'.
        embed_model_name (str): The name of the model used for embedding the text chunks.
        summ_model_name (str): The name of the model used for summarization.
        doc (str): The input document to be summarized.
        num_clusters (int): The number of clusters to form during clustering.
        top_k (int): The top k closest chunks to the cluster centroid.
        system_prompt_aggregate_summaries (str): System prompt used for generating aggregate summaries. requires how the doc starts 
        system_prompt_docsummary (str): System prompt used for generating the final document summary. requires how the doc starts 
        llm_instructions_doc_summary (str): Instructions for the final summarization task.
        reference_summary (str): A reference summary for evaluation purposes.
        client (OpenAI): An instance of the OpenAI client.
        log (bool, optional): Whether to log progress. Defaults to False.

    Returns:
        Dict[str, Any]: Results containing the generated summary and evaluation metrics including ROUGE, BERTScore, and coherence.

    """
    client = OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")

    if log:
        logger.info("Pipeline started")

    # Validate inputs
    assert chunking_method in CHUNKING_METHODS, "Invalid chunking method provided."
    if chunking_method == "recursive":
        assert "max_length" in chunking_params.keys() and "overlap" in chunking_params.keys(), \
            "'max_length' and 'overlap' must be provided for recursive chunking."
    elif chunking_method == "semantic":
        assert "threshold" in chunking_params.keys() and "model" in chunking_params.keys(), \
            "'threshold' and 'model' must be provided for semantic chunking."
    assert doc, "Document must not be None."

    if log:
        logger.info(f"Chunking method: {chunking_method}")
        logger.info(f"Embedding model: {embed_model_name}")
        logger.info(f"Summarization model: {summ_model_name}")
        logger.info("Parameters validated successfully")

    # Chunk the document based on the chosen method
    if chunking_method == 'recursive':
        words = doc.split()
        chunks, avg_chunk_length = iterative_chunk(words, **chunking_params)
        if log:
            logger.info(f"Document chunked using recursive method: {len(chunks)} chunks created")
    elif chunking_method == 'semantic':
        sentences = doc.split(".")
        chunks = semantic_chunking(sentences, **chunking_params)
        if log:
            logger.info(f"Document chunked using semantic method: {len(chunks)} chunks created")

    # Create vector database
    vector_database = create_vbd(chunks, emb_model=embed_model_name)
    if log:
        logger.info("Vector database created")

    # Perform clustering
    clustered_chunks, cluster_centroids , num_clusters= cluster_chunks_kmeans(vdb=vector_database)
    if log:
        logger.info(f"Clustering completed: {num_clusters} clusters formed")

    # Identify closest chunks to cluster centroids
    closest_chunks = find_closest_data_points_to_centroid(
        clustered_chunks=clustered_chunks,
        cluster_centroids=cluster_centroids,
        top_k=top_k
    )
    if log:
        logger.info(f"Identified top {top_k} closest chunks to cluster centroids")

    # Aggregate summaries for each cluster
    cluster_summaries = aggregate_summaries_for_each_cluster(
        closest_chunks,
        sys_prompt_content=system_prompt_aggregate_summaries, #has parameter called start -> how the document starts 
        model=summ_model_name,
        client=client
    )
    if log:
        logger.info("Generated aggregate summaries for each cluster")

    # Create transition matrix and directed graph
    transition_matrix = create_transition_matrix(clustered_chunks, n_clusters=num_clusters)
    graph = create_directed_graph(transition_matrix=transition_matrix, summary_by_cluster=cluster_summaries , plot = False)
    if log:
        logger.info("Transition matrix and directed graph created")

    path = list(range(num_clusters))
    if log:
        print(f"Path identified wo MarkovChain -> path : {path}")

    # Prepare messages for the final summary call
    messages = [{'role': 'system', 'content': system_prompt_docsummary}]
    for cluster_id in path:
        summary = cluster_summaries[cluster_id]
        messages.append({'role': 'user', 'content': summary})
    messages.append({"role": "user", 'content': llm_instructions_doc_summary})
    if log : 
        print(f"messages for generating the final summary are  : \n{messages}")
    # Generate the final summary
    llm = CollectionLLM.llm_collection[summ_model_name]
    overall_summary = llm.get_response(messages)
    if log:
        logger.info("Final document summary generated")

    # Calculate evaluation metrics
    candidates = [overall_summary]
    references = [[reference_summary]]

    if log:
        logger.info("Calculating ROUGE scores")
    rouge_results = calculate_rouge_scores(candidates, references)

    if log:
        logger.info("Calculating BERTScore")
    bert_results = calculate_bertscore(candidates, references)

    if log:
        logger.info("Calculating coherence scores")
    coherence_results = calculate_coherence(overall_summary)
    if log:
        logger.info("Calculating Blue Rt scores")
    blue_rt_scores = calculate_blue_rt_scores([overall_summary], [reference_summary])
    
    results = {
        'summary': overall_summary,
        'number clusters' : num_clusters , 
        'rouge': rouge_results,
        'bertscore': bert_results,
        'coherence': coherence_results,
        'blue_rt': blue_rt_scores
    }

    if log:
        logger.info("Pipeline completed")

    return results




