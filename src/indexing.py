import ollama 
from tqdm import tqdm
import time
import numpy as np 
from typing import List, Tuple, Union , Any , Dict 
import time
from functools import lru_cache

@lru_cache(maxsize=None)
def get_embedding_with_retry_ollama(
    model: Any, 
    prompt: str, 
    max_retries: int = 5, 
    initial_delay: int = 1,
    prefix = 'clustering : '
    ) -> List[float]:
    """
    Attempts to get an embedding from the Ollama model with retry capability on failure.

    Args:
        model (Any): The model object or identifier used to generate embeddings from Ollama.
        prompt (str): The input string for which to generate the embedding.
        max_retries (int, optional): The maximum number of retry attempts in case of failure. Defaults to 5.
        initial_delay (int, optional): The initial delay between retries in seconds. Defaults to 1.

    Returns:
        List[float]: The embedding vector corresponding to the input prompt.

    Raises:
        ollama.ResponseError: If the maximum number of retries is reached and the function fails to get the embedding.
    """
    retries = 0
    delay = initial_delay

    while retries < max_retries:
        try:
            response = ollama.embeddings(model=model, prompt=prefix + prompt)
            return response["embedding"]
        except ollama.ResponseError as e:
            retries += 1
            print(f"Failed to get embedding. Attempt {retries}/{max_retries}. Error: {e}")
            if retries >= max_retries:
                print(f"Max retries reached for sentence: {prompt}.")
                raise e
            time.sleep(delay)
            delay *= 2  # Exponential backoff





def join_words_into_text(words: List[str]) -> str:
    """Joins a list of words into a single text string."""
    return ' '.join(words)

def split_text_into_words(text: str) -> List[str]:
    """Splits a text string into a list of words."""
    return text.split()

def iterative_chunk(
    words: List[str], 
    max_length: int, 
    overlap: int
    ) -> Tuple[Union[List[str], List[float]], float]:
    """
    Splits a list of words into chunks with a specified maximum length and overlap.

    Args:
        words (List[str]): The list of words to be chunked.
        max_length (int): The maximum number of words allowed in each chunk.
        overlap (int): The number of words that should overlap between consecutive chunks.

    Returns:
        Tuple[Union[List[str], List[float]], float]: A tuple where the first element is a list of text chunks (or a list with a single element as float('inf') if overlap is greater than or equal to max_length), and the second element is the average chunk length in number of words (or float('inf') if there's an error).
    """
    # Check if overlap is greater than or equal to max_length
    if overlap >= max_length:
        return [float('inf')], float('inf')

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_length, len(words))
        chunk = words[start:end]
        chunks.append(join_words_into_text(chunk))
        start += max_length - overlap

    avg_chunk_length = np.mean([len(split_text_into_words(chunk)) for chunk in chunks])
    return chunks, avg_chunk_length


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embeddings_for_sentences(model: Any, sentences: List[str]) -> List[Dict[str, Any]]:
    """
    Generates embeddings for a list of sentences.

    Args:
        model (Any): The model object or identifier used to generate embeddings from Ollama.
        sentences (List[str]): The list of sentences for which to generate embeddings.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing a 'sentence' and its corresponding 'embedding'.
    """
    data = []
    for sentence in tqdm(sentences, desc="Generating embeddings"):
        try:
            embedding = get_embedding_with_retry_ollama(model, sentence , prefix = 'classification : ')
            data.append({"sentence": sentence, "embedding": embedding})
        except ollama.ResponseError as e:
            print(f"Failed to generate embedding for sentence: {sentence}. Error: {e}")
            data.append({"sentence": sentence, "embedding": []})  # Use empty list for failed embeddings
    return data

def semantic_chunking(sentences: List[str], model: Any, threshold: float) -> List[str]:
    """
    Groups sentences into semantic chunks based on the cosine similarity of their embeddings.

    Args:
        sentences (List[str]): A list of sentences to be chunked.
        model (Any): The model object or identifier used to generate embeddings from Ollama.
        threshold (float, optional): The similarity threshold to determine if two embeddings are similar enough to be in the same chunk. Defaults to 0.8.

    Returns:
        List[str]: A list of chunks, where each chunk is a string of semantically grouped sentences.
    """
    data = get_embeddings_for_sentences(model, sentences)
    chunks = []
    i = 0

    while i < len(data) - 1:
        a = data[i]["embedding"]
        b = data[i + 1]["embedding"]

        # Check for empty embeddings
        if len(a) == 0 or len(b) == 0:
            chunks.append(data[i]["sentence"])
            i += 1
            continue

        sim = cosine_similarity(a, b)
        chunk = data[i]["sentence"]

        while i < len(data) - 2:
            a = data[i]["embedding"]
            b = data[i + 1]["embedding"]

            # Check for empty embeddings
            if len(a) == 0 or len(b) == 0:
                break

            sim = cosine_similarity(a, b)
            if sim < threshold:
                break

            chunk += " " + data[i + 1]["sentence"]
            i += 1

        chunks.append(chunk)
        i += 1

    # If the last sentence hasn't been added, append it as its own chunk
    if i == len(data) - 1:
        chunks.append(data[i]["sentence"])

    return chunks



def create_vbd(chunks, emb_model)-> List[Dict[str ,Any]]:
    """
    Creates a vector database from a list of chunks using a specified embedding model.
    
    This function processes a list of text chunks, generates embeddings for each chunk using 
    the provided embedding model, and stores the results in a vector database. The vector 
    database is represented as a list of dictionaries, where each dictionary contains:
    
    Attributes:
    ----------
    - `pos` : int
        The chronological position of the chunk in the original document.
    - `chunk` : str
        The content of the chunk extracted from the document.
    - `embedding` : numpy.ndarray
        The embedding of the chunk as generated by the embedding model.
    
    Parameters:
    ----------
    chunks : list of str
        A list of text chunks extracted from the document that will be embedded.
        
    emb_model : object
        The embedding model used to generate embeddings for the chunks. This should be 
        compatible with the `get_embedding_with_retry_ollama` function which expects a model 
        and a prompt (chunk) to generate embeddings.
    
    Returns:
    -------
    list of dict
        A list of dictionaries, each containing the position, chunk, and its corresponding 
        embedding as described above.
    
    Raises:
    ------
    Any exceptions raised by `get_embedding_with_retry_ollama`, such as:
        - Network-related issues if embeddings are fetched via an API.
        - Model-specific errors if the model fails to generate embeddings.
    
    Note:
    ----
    - If an embedding is empty (i.e., `emb.shape == (0,)`), the chunk is skipped and not 
      added to the vector database.
    - The function uses tqdm for progress tracking, showing progress as chunks are processed.
    
    Example:
    -------
    >>> chunks = ["This is the first chunk.", "This is the second chunk."]
    >>> emb_model = "example_model"
    >>> vdb = create_vbd(chunks, emb_model)
    >>> print(vdb)
    [{'pos': 0, 'chunk': 'This is the first chunk.', 'embedding': array([...])}, 
     {'pos': 1, 'chunk': 'This is the second chunk.', 'embedding': array([...])}]
    """
    
    vdb = []
    i = 0 
    for chunk in tqdm(chunks, desc="embedding chunks..."):
        emb = np.array(get_embedding_with_retry_ollama(model=emb_model, prompt=chunk)) #embeddings
        if not emb.shape == (0,): 
            dp = {
                "pos": i, # chronological position of the chunk 
                "chunk": chunk, # the chunk itself 
                "embedding": emb
            }
            i += 1
            vdb.append(dp)
    
    return vdb
