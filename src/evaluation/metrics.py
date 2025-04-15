import spacy
import textdescriptives as td
import evaluate
import bert_score
from bleurt import score
from typing import List 
rouge = evaluate.load('rouge')
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textdescriptives/coherence")
checkpoint = "BLEURT-20"
scorer = score.BleurtScorer(checkpoint)

def calculate_blue_rt_scores(candidates : List[str], references : List[str]) -> List[float]:
    """
    Calculate BLEU scores for given candidates and references.

    Args:
        candidates (List[str]): list of candidate summaries.
        references (List[str]): list of reference summaries.
    Returns:
        List[float]: BLEU scores for each candidate.
    """
    scores = scorer.score(references=references, candidates=candidates)
    return scores

def calculate_rouge_scores(candidates, references):
    """
    Calculate ROUGE scores for given candidates and references.

    Args:
        candidates (list of str): List of candidate summaries.
        references (list of list of str): List of reference summaries for each candidate.

    Returns:
        dict: ROUGE scores including ROUGE-1, ROUGE-2, ROUGE-L.
    """
   
    results = rouge.compute(predictions=candidates, references=references)
    refined_results = {}
    for rouge_k , score  in results.items() : 
        refined_results[rouge_k] = 100 * score
    return refined_results

def calculate_bertscore(candidates, references):
    """
    Calculate BERTScore for given candidates and references.

    Args:
        candidates (list of str): List of candidate summaries.
        references (list of list of str): List of reference summaries for each candidate.

    Returns:
        dict: BERTScore including Precision, Recall, and F1.
    """
    # Ensure the number of candidates matches the number of sets of reference summaries
    assert len(candidates) == len(references), "Different number of candidates and sets of references"

    # Flatten the list of references for each candidate
    flat_references = [' '.join(ref) for ref in references]

    # Compute BERTScore
    P, R, F1 = bert_score.score(candidates, flat_references, lang='en', verbose=True)

    # Average BERTScore values for candidates
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    return {
        'bert_precision': avg_precision,
        'bert_recall': avg_recall,
        'bert_f1': avg_f1
    }


def calculate_coherence(text):
    """
    Calculate first-order and second-order coherence for a given text.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: Coherence scores including first-order and second-order coherence.
    """


    doc = nlp(text)
    coherence = doc._.coherence
    first_order_coherence = doc._.first_order_coherence_values
    second_order_coherence = doc._.second_order_coherence_values

    return {
        'first_order_coherence': coherence['first_order_coherence'],
        'second_order_coherence': coherence['second_order_coherence'],
        'first_order_values': first_order_coherence,
        'second_order_values': second_order_coherence
    }

if __name__ == "__main__" : 
    # Test calculate_rouge_scores
    candidates = ["The cat is on the mat.", "There is a cat on the mat."]
    references = [["The cat is on the mat.", "A cat is on the mat."], ["There is a cat on the mat.", "A cat is on the mat."]]
    rouge_scores = calculate_rouge_scores(candidates, references)
    print("ROUGE Scores:", rouge_scores)

    # Test calculate_bertscore
    bertscore = calculate_bertscore(candidates, references)
    print("BERTScore:", bertscore)

    # Test calculate_coherence
    text = "Cats are wonderful pets. They purr and lie on your lap. A cat can be a great companion."
    coherence_scores = calculate_coherence(text)
    print("Coherence Scores:", coherence_scores)