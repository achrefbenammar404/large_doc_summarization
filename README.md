<h1 align="center">Voyverse — Markov-Enhanced Long Document Summarization</h1>
<p align="center">
  <img src="https://www.voyverse.com/logo.png" alt="Voyverse Logo" width="270"/>
  <img src="https://insat.rnu.tn/assets/images/logo_c.png" alt="INSAT logo" width ="100">
</p>
<p align="center">
 <img src="https://ifipaiai.org/wp-content/uploads/2021/10/aiai_site_logo-1.png" alt="INSAT logo" width ="280">
</p>


<p align="center">
  Official implementation of the research paper:<br>
  <strong>"Markov-Enhanced Clustering for Long Document Summarization: Tackling the 'Lost in the Middle' Challenge with Large Language Models"</strong><br>
  by <em>Aziz Amari</em> and <em>Mohamed Achref Ben Ammar</em>, INSAT, University of Carthage, Tunisia.
</p>

---

## Abstract

As the volume of information continues to grow across domains, the need for accurate and scalable automatic text summarization methods has become increasingly critical. While large language models (LLMs) have significantly advanced abstractive summarization, they remain challenged by the "lost in the middle" problem — the inability to effectively capture and retain critical information in lengthy documents.

This repository presents a hybrid summarization framework that integrates extractive and abstractive techniques. The method involves chunking the input text, embedding and clustering the segments, generating summaries for each cluster, and constructing a final coherent summary based on a Markov chain representation of semantic transitions between clusters. This graph-based ordering mechanism enhances logical flow and relevance in the generated summary.

---

## Methodology

The summarization pipeline consists of the following key components:

1. **Chunking**  
   The input document is segmented into smaller, semantically meaningful units.

2. **Embedding**  
   Each chunk is converted into a high-dimensional vector representation using pre-trained embedding models.

3. **Clustering**  
   Chunks are grouped based on similarity in the embedding space to identify underlying topics.

4. **Cluster Refinement**  
   Outliers are removed and reclustering is performed to ensure coherent groupings.

5. **Centroid-Based Selection**  
   Chunks closest to each cluster's centroid are selected as representative passages.

6. **Abstractive Summarization of Clusters**  
   Selected chunks are summarized using an LLM to capture each cluster’s key idea.

7. **Cluster Labeling**  
   Each cluster is assigned a semantic label based on the output summaries.

8. **Markov Transition Graph Construction**  
   A transition matrix is created to model the semantic flow between clusters.

9. **Ranking and Path Inference**  
   The most coherent sequence of clusters is determined via graph traversal.

10. **Final Summary Generation**  
    The selected path is used to generate the final abstractive summary of the entire document.

---

## Installation and Setup

To reproduce the results and run the summarization pipeline locally, follow the steps below:

### 1. Clone the Repository with Submodules

This repository includes [BLEURT](https://github.com/google-research/bleurt) as a submodule for evaluation purposes.

```bash
git clone --recursive https://github.com/voyverse/LargeDocSum.git
```

If you cloned the repository without `--recursive`, initialize the submodule manually:

```bash
cd LargeDocSum
git submodule update --init --recursive
```

### 2. Install Ollama and Pull the Embedding Model

This project utilizes the `nomic-embed-text` model via [Ollama](https://ollama.com/):

- Download and install Ollama: https://ollama.com/download  
- Pull the required model:

```bash
ollama pull nomic-embed-text
```

### 3. Create a Virtual Environment and Install Dependencies

```bash
cd LargeDocSum
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 4. Install BLEURT

```bash
cd bleurt
pip install .
```

### 5. Download BLEURT Checkpoint

```bash
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
```

### 6. Configure Environment Variables

Create a `.env` file in the root directory with the following entries:

```env
GPT_4O_MINI_AZURE_TARGET_URI=your_azure_uri_here
GPT_4O_MINI_AZURE_API_KEY=your_api_key_here
GPT_4O_MINI_AZURE_API_VERSION=your_api_version_here
ENVIRONMENT="local"
```

### 7. Run the Pipeline

Once all dependencies are set up, you can execute the summarization pipeline on your documents.

---

## Usage

This system is designed for generating coherent summaries of long-form documents across a variety of domains such as legal texts, scientific literature, and technical manuals. It is particularly suitable for applications where both topical coverage and logical flow are critical.

### Quick Start (Python & Streamlit)

#### Using the Summarization Pipeline in Python

Once you've installed the dependencies and setup the environment, you can easily integrate the summarization pipeline into your own scripts:

```python
from src import summarization_pipeline
from openai import OpenAI

# Load document (replace with your file loading logic)
with open("example_document.txt", "r", encoding="utf-8") as f:
    document_text = f.read()

# Create the OpenAI-compatible client (Ollama or your Azure endpoint)
client = OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")

# Run the summarization pipeline
summary_output = summarization_pipeline.pipeline(
    chunking_method="recursive",
    chunking_params={"max_length": 500, "overlap": 20},
    embed_model_name="nomic-embed-text",
    summ_model_name="gpt-4o-mini",
    doc=document_text,
    top_k=20,
    system_prompt_aggregate_summaries="You are an expert AI writing assistant...",
    system_prompt_docsummary="You are an expert writing assistant...",
    llm_instructions_doc_summary="Combine the provided chapters into a single document...",
    reference_summary=None,  # Set this if you want to compute metrics
    client=client,
    log=True
)

# Print the final summary
print(summary_output["summary"])
```

This will return a dictionary with the following keys:
- `"summary"` — the final generated summary
- `"rouge"`, `"bertscore"` — evaluation metrics (if a reference summary is provided)
- `"coherence"` — internal coherence score
- `"blue_rt"` — estimated real-time response duration

---

#### Running the Streamlit App

To test the interactive summarization demo:

1. Install Streamlit if you haven't already:
   ```bash
   pip install streamlit
   ```

2. Run the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

The Streamlit app allows you to upload documents, select chunking strategies, adjust parameters, and view summaries and metrics with a clean UI.

## Citation

If you use this codebase in your research, please cite the following paper:

```bibtex
@inproceedings{amaribenammar2025markov,
  title     = {Markov-Enhanced Clustering for Long Document Summarization: Tackling the 'Lost in the Middle' Challenge with Large Language Models},
  author    = {Aziz Amari and Mohamed Achref Ben Ammar},
  booktitle = {Proceedings of AIAI 2025 - 21st International Conference on Artificial Intelligence Applications and Innovations - Cyprus University of Technology, Limassol, Cyprus},
  year      = {2025},
  institution = {National Institute of Applied Science and Technology (INSAT), University of Carthage}
}
```

---

## Contact

For questions, feedback, or collaboration inquiries, please contact:

- Aziz Amari: [aziz.amari@insat.ucar.tn](mailto:aziz.amari@insat.ucar.tn)  
- Mohamed Achref Ben Ammar: [mohamedachref.benammar@insat.ucar.tn](mailto:mohamedachref.benammar@insat.ucar.tn)
