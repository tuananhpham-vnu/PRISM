
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from config import MODEL_CACHE_PATH
from helper import (LLAMA2_7B, clean_name, downstream_folder_name, downstream_tasks, RBPO, RMEPO, GEMMA_EMBEDDING_MODEL, MINILM_EMBEDDING_MODEL, base_llm_models, BBH, device, embedding_models)
import json, os

from config import prompt_template_gsm8k, prompt_template_piqa, prompt_template_multiple_choice
from utils import generate, generate_batch
from utils_downstream_task import format_prompt_template

load_dotenv()
hf_token = os.getenv("HF_TOKEN")    

# embedding_models = [
#     GEMMA_EMBEDDING_MODEL,
#     MINILM_EMBEDDING_MODEL
# ]

base_llm_models = [LLAMA2_7B]

METHOD = [
    RBPO,
    RMEPO
]

res = []
for method in METHOD:
    for base_model in base_llm_models:
        for embed_model_name in embedding_models:
            for task in downstream_tasks:
                if task == "demo":
                    continue
                if task == BBH:
                    multiple_choice = [
                    'date_understanding', 'disambiguation_qa', 'hyperbaton', 'logical_deduction_five_objects',
                    'logical_deduction_seven_objects', 'logical_deduction_three_objects',
                    'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects',
                    'ruin_names', 'salient_translation_error_detection', 'snarks',
                    'temporal_sequences', 'tracking_shuffled_objects_five_objects',
                    'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
                    'causal_judgement','formal_fallacies','navigate','web_of_lies',
                    'sports_understanding','boolean_expressions',
                    'multistep_arithmetic_two', 'object_counting', 'word_sorting'
                ]
                    for subtask in multiple_choice:
                        data_path = f"{downstream_folder_name}/{method}/{clean_name(embed_model_name)}/{task}/{subtask}.json"
                        with open(data_path, "r", encoding="utf-8") as f:
                            data = json.load(f)[:5] # NOTE: chỉ lấy 5 sample đầu để test
                        print(f"\nBase model: {base_model} | Task: {subtask} | Embedding: {clean_name(embed_model_name)} | Dataset: {data_path} | Loaded {len(data)} samples")                        
                        
                        output_path = f"{downstream_folder_name}/{method}/{clean_name(embed_model_name)}/{task}/{subtask}_demo.json"
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                else:
                    data_path= f"{downstream_folder_name}/{method}/{clean_name(embed_model_name)}/{task}.json"

                    with open(data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)[:5] # NOTE: chỉ lấy 5 sample đầu để test
                    output_path = f"{downstream_folder_name}/{method}/{clean_name(embed_model_name)}/{task}_demo.json"
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)     