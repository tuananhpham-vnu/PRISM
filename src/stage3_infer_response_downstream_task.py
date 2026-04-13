
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from config import MODEL_CACHE_PATH
from helper import (LLAMA2_7B, clean_name, downstream_folder_name, downstream_task_datasets, RBPO, RMEPO, GEMMA_EMBEDDING_MODEL, MINILM_EMBEDDING_MODEL, base_llm_models, BBH, device, embedding_models)
import json, os

from config import prompt_template_gsm8k, prompt_template_piqa, prompt_template_multiple_choice

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
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                cache_dir=MODEL_CACHE_PATH,
                token = hf_token,
                torch_dtype="auto"
            ).eval().to(device)

            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                cache_dir=MODEL_CACHE_PATH,
                token = hf_token,
                legacy=False
            )
            for task in downstream_task_datasets:
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
                        data_path = f"{downstream_folder_name}/{method}/{clean_name(embed_model_name)}/{task}/{subtask}_cluster.json"
                        with open(data_path, "r", encoding="utf-8") as f:
                            data = json.load(f)[:2]
                        print(f"\nBase model: {base_model} | Task: {subtask} | Dataset: {data_path} | Loaded {len(data)} samples")
                        
                        # sua batch cho doan nay di, doan nay call api thi nen batching sang gpu duoc khong?
                        for item in data:
                            prompt = item.get(f"{method}_prompt", "")
                            
                        
                            
                        # obj = f"Loaded data for {method} - {embed_model_name} - {task} - 
                        # {subtask}, number of samples: {len(data)}"                        
                        # res.append(obj)
                else:
                    data_path= f"{downstream_folder_name}/{method}/{clean_name(embed_model_name)}/{task}_cluster.json"

                    with open(data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)[:2]
                    print(f"\nBase model: {base_model} | Task: {task} | Dataset: {data_path} | Loaded {len(data)} samples")
                    
                    # obj = f"Loaded data for {method} - {embed_model_name} - {task}, number of samples: {len(data)}"
                    # res.append(obj)
        if os.path.exists(MODEL_CACHE_PATH):
            import shutil
            shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)           

# with open("data_overview.txt", "w", encoding="utf-8") as f:
#     for line in res:
#         f.write(line + "\n")