import streamlit as st
import tempfile
import os
import json
from src import summarization_pipeline
from openai import OpenAI
from app.utils import extract_file_text
from src.logger.logger import get_logger
import asyncio
import torch 
torch.classes.__path__ = [] 

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

logger = get_logger(__file__)
st.set_page_config(page_title="Document Summarizer", layout="wide" , page_icon="📝")
# Centered logo and title
st.markdown("""<p align="center">
            Official implementation of the research paper:<br>
            <strong>"Markov-Enhanced Clustering for Long Document Summarization: Tackling the 'Lost in the Middle' Challenge with Large Language Models"</strong><br>
            by <em>Aziz Amari</em> and <em>Mohamed Achref Ben Ammar</em>, INSAT, University of Carthage, Tunisia.
            </p>
""", unsafe_allow_html=True)
st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; margin-bottom: 1rem;">
        <img src="https://www.voyverse.com/logo.png" width="300">
        <img src="https://insat.rnu.tn/assets/images/logo_c.png" width="100">
    </div>
""", unsafe_allow_html=True)


# Custom styling to match Geist-like fonts and a clean layout
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Geist', sans-serif;
            background-color: #f9f9f9;
            color: #333;
        }

        .main > div {
            padding-top: 2rem;
        }

        .block-container {
            padding: 2rem 2rem;
        }

        .title-style {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)
# Description
st.markdown("""
### Overview
This application showcases the implementation of our research methodology presented in:

**Markov-Enhanced Clustering for Long Document Summarization: Tackling the 'Lost in the Middle' Challenge with Large Language Models**  
*Aziz Amari, Mohamed Achref Ben Ammar*  
National Institute of Applied Science and Technology (INSAT), University of Carthage, Tunis, Tunisia

> A hybrid summarization approach that combines extractive and abstractive techniques using clustering and Markov chains for semantic ordering of ideas.

---
""")
# Citation section
st.markdown("""
    ---
    ### Citation
    If you use this app or find our work useful, please cite:

    **Aziz Amari, Mohamed Achref Ben Ammar**  
    *Markov-Enhanced Clustering for Long Document Summarization: Tackling the 'Lost in the Middle' Challenge with Large Language Models*  
    National Institute of Applied Science and Technology (INSAT), University of Carthage, Tunis, Tunisia  
    ```bibtex
    @inproceedings{
        amaribenammar2025markov,
        title     = {Markov-Enhanced Clustering for Long Document Summarization: Tackling the 'Lost in the Middle' Challenge with Large Language Models},
        author    = {Aziz Amari and Mohamed Achref Ben Ammar},
        booktitle = {Proceedings of AIAI 2025 - 21st International Conference on Artificial Intelligence Applications and Innovations - Cyprus University of Technology, Limassol, Cyprus},
        year      = {2025},
        institution = {National Institute of Applied Science and Technology (INSAT), University of Carthage}
    }
    ```

    > A hybrid summarization approach combining extractive and abstractive techniques with Markov chains for semantic reordering of ideas.
""")



# Sidebar configuration with logo
st.sidebar.image("https://www.voyverse.com/logo.png", width=150)
st.sidebar.header("Configuration")
chunking_method = st.sidebar.selectbox("Chunking Method", ["recursive", "semantic"])
max_length = st.sidebar.slider("Max Chunk Length", min_value=100, max_value=2000, value=500)
overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=200, value=20)
top_k = st.sidebar.slider("Top-k Chunks", min_value=1, max_value=50, value=20)
model_name = st.sidebar.text_input("Summarization Model", "gpt-4o-mini")

# Upload file
uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT, etc.)", type=["pdf", "txt", "md", "tex"])
reference_summary = st.text_area("Optional: Paste a reference summary here for evaluation", "")
reference_summary = reference_summary if reference_summary.strip() else []



if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("File uploaded successfully.")

    try:
        doc_text = extract_file_text(tmp_path, uploaded_file.name)
        st.subheader("📄 Extracted Document Preview")
        st.text_area("", doc_text[:2000] + ("..." if len(doc_text) > 2000 else ""), height=300)

        if st.button("🔍 Summarize Document"):
            st.info("Running summarization pipeline...")

            client = OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")

            # Prompts
            system_prompt_aggregate_summaries = (
                "You are an expert AI writing assistant specializing in synthesizing and rewriting content into cohesive and detailed summaries..."
            )
            system_prompt_docsummary = (
                "You are an expert writing assistant specializing in document synthesis and content generation..."
            )
            llm_instructions_doc_summary = (
                "Combine the provided chapters into a single document. Ensure that the content flows logically and maintains the overall narrative..."
            )

            output = summarization_pipeline.pipeline(
                chunking_method=chunking_method,
                chunking_params={"max_length": max_length, "overlap": overlap},
                embed_model_name="nomic-embed-text",
                summ_model_name=model_name,
                doc=doc_text,
                top_k=top_k,
                system_prompt_aggregate_summaries=system_prompt_aggregate_summaries,
                system_prompt_docsummary=system_prompt_docsummary,
                llm_instructions_doc_summary=llm_instructions_doc_summary,
                reference_summary=reference_summary,
                client=client,
                log=True
            )

            st.subheader("📘 Generated Summary")
            st.write(output['summary'])

            st.subheader("📊 Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**ROUGE Scores:**")
                st.json(output['rouge'])
                st.write("**BERTScore:**")
                st.json(output['bertscore'])
            with col2:
                st.write("**Coherence Scores:**")
                st.json(output['coherence'])
                st.write("**Blue_RT:**")
                st.write(output['blue_rt'])

    except Exception as e:
        st.error(f"An error occurred while processing the document: {e}")
    finally:
        os.remove(tmp_path)
else:
    st.info("Please upload a document to begin summarization.")
