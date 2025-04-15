import pandas as pd
import json
import os
from src import summarization_pipeline
from src import wo_markov_summarization_pipeline
from src import pure_llm_approach 
from src.data.hf_cleaned_booksum import DataLoader
from openai import OpenAI
from src.logger.logger import get_logger

logger = get_logger(__file__)

data_set = 'test'
# Initialize data loader
data_loader = DataLoader(data_set=data_set)
number_of_books = data_loader.get_number_of_books()
logger.debug(f"Number of Books to summarize is {number_of_books}")
# Initialize OpenAI client
client = OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")

# Define common parameters
chunking_method = 'recursive'  # or 'semantic', depending on your requirement
chunking_params = {
    'max_length': 500,  # example value
    'overlap': 20      # example value
}
embed_model_name = "nomic-embed-text"
summ_model_name = "gpt-4o-mini"
top_k = 20  # example value

# Directory to save JSON files
our_approach_output_dir = os.path.join('output' , 'booksum' ,"our_approach" , f'summarization_results_{data_set}')
llm_approach_output_dir = os.path.join('output' , 'booksum' , "llm_approach" , f'summarization_results_{data_set}')
wo_markov_approach_output_dir = os.path.join('output' , 'booksum' , "wo_markov_approach" , f'summarization_results_{data_set}')


os.makedirs(our_approach_output_dir, exist_ok=True)
os.makedirs(llm_approach_output_dir, exist_ok=True)
os.makedirs(wo_markov_approach_output_dir, exist_ok=True)

# List to collect results for DataFrame
our_approach_results = []
llm_pure_approach_results = []
wo_markov_chain_approach_results = []


for index_to_fetch in range(number_of_books):
    
    try :
        book_summary_couple = data_loader.get_book_summary_couple(index_to_fetch)
        doc = book_summary_couple["book"]
        reference_summary = book_summary_couple["summary"]

        # =====================================
        #========OUR APPROACH ================

        # Define prompts using the document content
        system_prompt_aggregate_summaries = (
        "You are an expert AI writing assistant specializing in synthesizing and rewriting content into cohesive and detailed summaries. Your task is to aggregate multiple chunks from the same chapter or part of a document, ensuring all key points, details, insights, and nuances are preserved. As you rewrite the content, maintain the original intent, depth, and lexicon wherever appropriate, and present the material as a fluent, extended summary that reads as a complete and cohesive narrative. Your goal is to ensure that no critical information is lost and the rewritten content aligns seamlessly with the original text’s style and purpose."
        )
        system_prompt_docsummary = (
        "You are an expert writing assistant specializing in document synthesis and content generation. I will provide you with several chapters from a longer document. Your task is to integrate these chapters into a single, cohesive document that retains all critical information, themes, and insights from the original content. Ensure that the resulting document flows logically and maintains the narrative style of the source material. Pay close attention to preserving the integrity of key details and structuring the output in a way that mirrors the original's intent and style. The final document should be thorough, comprehensive, and reflective of the complete set of chapters provided."
        )

        llm_instructions_doc_summary = 'Combine the provided chapters into a single document. Ensure that the content flows logically and maintains the overall narrative style of the original chapters.The final document should be thorough and extensive, encapsulating the entire content of the provided chapters. It should give a complete and clear representation of the material. Retain all essential information, themes, and insights from the original chapters. Do not omit or alter critical details that could affect the understanding or intent of the original content. b'

        # Run the pipeline
        output = summarization_pipeline.pipeline(
            chunking_method=chunking_method,
            chunking_params=chunking_params,
            embed_model_name=embed_model_name,
            summ_model_name=summ_model_name,
            doc=doc,
            top_k=top_k,
            system_prompt_aggregate_summaries=system_prompt_aggregate_summaries,
            system_prompt_docsummary=system_prompt_docsummary,
            llm_instructions_doc_summary=llm_instructions_doc_summary,
            reference_summary=reference_summary,
            client=client,
            log=True  
        )

        result = {
            'Book Index': index_to_fetch,
            'Reference Summary' : reference_summary , 
            'number clusters' : output["number clusters"] ,
            'Summary': output['summary'],
            'ROUGE Scores': output['rouge'],
            'BERTScore': output['bertscore'],
            'Coherence Scores': output['coherence'],
            'blue_rt': output['blue_rt']
        }

        json_filename = os.path.join(our_approach_output_dir, f'book_summary_{index_to_fetch}.json')
        with open(json_filename, 'w') as json_file:
            json.dump(result, json_file, indent=4)

        our_approach_results.append(result)

        
        
        # =====================================
        #========PURE LLM APPROACH ============
        summ_system_prompt = "you are an excellent summarization agent"
        llm_context_window = 128000
        output = pure_llm_approach.pipeline(summ_model_name="gpt-4o-mini" , doc = doc , reference_summary=reference_summary , summ_sys_prompt=summ_system_prompt , llm_context_window=llm_context_window , log = True )
        result = {
            'Book Index': index_to_fetch,
            'Reference Summary' : reference_summary , 
            'Summary': output['summary'],
            'ROUGE Scores': output['rouge'],
            'BERTScore': output['bertscore'],
            'Coherence Scores': output['coherence'],
            'blue_rt': output['blue_rt']
        }
        json_filename = os.path.join(llm_approach_output_dir, f'book_summary_{index_to_fetch}.json')
        with open(json_filename, 'w') as json_file:
            json.dump(result, json_file, indent=4)

        llm_pure_approach_results.append(result)
        
        
        # ======================================
        #========WO MARKOV CHAIN APPROACH ======
        
        output = wo_markov_summarization_pipeline.pipeline(
            chunking_method=chunking_method,
            chunking_params=chunking_params,
            embed_model_name=embed_model_name,
            summ_model_name=summ_model_name,
            doc=doc,
            top_k=top_k,
            system_prompt_aggregate_summaries=system_prompt_aggregate_summaries,
            system_prompt_docsummary=system_prompt_docsummary,
            llm_instructions_doc_summary=llm_instructions_doc_summary,
            reference_summary=reference_summary,
            client=client,
            log=True  
        )

        result = {
            'Book Index': index_to_fetch,
            'Reference Summary' : reference_summary , 
            'number clusters' : output["number clusters"] ,
            'Summary': output['summary'],
            'ROUGE Scores': output['rouge'],
            'BERTScore': output['bertscore'],
            'Coherence Scores': output['coherence'],
            'blue_rt': output['blue_rt']
        }

        json_filename = os.path.join(wo_markov_approach_output_dir, f'book_summary_{index_to_fetch}.json')
        with open(json_filename, 'w') as json_file:
            json.dump(result, json_file, indent=4)

        wo_markov_chain_approach_results.append(result)
    except Exception as e : 
        logger.error(f"Error processing book index {index_to_fetch}: {e}")
        continue
    

df = pd.DataFrame(our_approach_results)
df.to_csv(os.path.join(our_approach_output_dir , f'our_approach_summarization_results_{data_set}.csv'), index=False)


df = pd.DataFrame(llm_pure_approach_results)
df.to_csv(os.path.join(llm_approach_output_dir ,f'llm_pure_approach_summarization_results_{data_set}.csv'), index=False)


df = pd.DataFrame(wo_markov_chain_approach_results)
df.to_csv(os.path.join(wo_markov_approach_output_dir ,f'wp_markov_chain__approach_summarization_results_{data_set}.csv') ,  index=False)



