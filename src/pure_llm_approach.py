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
from src.llm.llm import CollectionLLM
from transformers import AutoTokenizer

logger = get_logger(__file__)


def chunk_text(text: str, max_tokens: int) -> List[str]:
    return  " ".join(text.split()[ : int(0.6 * max_tokens)])



def pipeline(
    summ_model_name: str,
    doc: str,
    reference_summary : str , 
    summ_sys_prompt : str ,
    llm_context_window : int,
    log: bool = False,
) -> Dict[str, Any]:
    """
    A summarization pipeline that processes a document through chunking, embedding, clustering,
    summarizing, and evaluating to generate a comprehensive summary.

    Args:
        summ_model_name (str): The name of the model used for summarization.
        doc (str): The input document to be summarized.
        summ_sys_prompt (str): System prompt used for generating the final document summary. 
        reference_summary (str): A reference summary for evaluation purposes.
        log (bool, optional): Whether to log progress. Defaults to False.
        llm_context_window (int) : context window length of the llm used 
    Returns:
        Dict[str, Any]: Results containing the generated summary and evaluation metrics including ROUGE, BERTScore, and coherence.

    """
    try : 
        llm = CollectionLLM.llm_collection[summ_model_name]
    except KeyError as e : 
        logger.error(f"Model {summ_model_name} doesn't exist in LLM Collection. ")
        return f"Model {summ_model_name} doesn't exist in LLM Collection. "
    
    chunk = chunk_text(doc,llm_context_window)
    
    overall_summary = llm.get_response(messages = [
        {"role" : "system" , "content" : summ_sys_prompt} , 
        {"role" : "user" , "content" : f"Here is the document to summarize : {chunk}"}
    ])
    
    
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
        'rouge': rouge_results,
        'bertscore': bert_results,
        'coherence': coherence_results,
        'blue_rt': blue_rt_scores
    }

    if log:
        logger.info("Pipeline completed")

    return results
    