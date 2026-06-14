import torch, os, gc, json

from tqdm import tqdm

from helper import DEEPSEEK, LLAMA2_7B, MEPO, MINILM_EMBEDDING_MODEL, RMEPO, SELF_INSTRUCT_EVAL, create_combined_name, device, experiment_file_name, M, eval_folder_name, set_global_seed, evaluator_models, DOLLY_EVAL,DEMO_EVAL
from config import MODEL_CACHE_PATH, SEED, prompt_template_optimize
from inference_batch import merge_optim_prompt_with_original, process_mepo_batch
from utils import generate_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mepo_inference import MePOModel


# =========================
# Step 0: Configure and load MePO model
# =========================
set_global_seed(SEED)
mepo_model = MePOModel()
batch_size = 8

# only handle ablation study for 3 phase
folder_name = f'ablation'
embedding_models = [MINILM_EMBEDDING_MODEL]
evaluation_datasets = [
    DOLLY_EVAL, SELF_INSTRUCT_EVAL
    # DEMO_EVAL
    ]

evaluator_models = [DEEPSEEK]
METHOD = [RMEPO]
base_llm_models = [LLAMA2_7B]

def prepare_ablation_file(source_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = []
    for idx, item in enumerate(data, start=1):
        normalized.append({
            "id": item.get("id") or item.get("question_id") or item.get("idx") or idx,
            "ori_prompt": item.get("instruction") or item.get("prompt") or item.get("text"),
            "context": item.get("context"),
            "bpo_prompt": item.get("optimized_prompt"),
            "category": item.get("category"),
            "expected_response": item.get("output") or item.get("good_res") or item.get("response"),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

# =========================
# Step 1: Infer MePO prompt for all items
# =========================
for base_llm in base_llm_models:
    for data_path in evaluation_datasets:
        for evaluator in evaluator_models:
            source_path = f'testset/{data_path}'
            output_path = f'{folder_name}/{create_combined_name(base_llm, data_path, evaluator)}.json'
            print(f"Processing: {output_path}")

            prepare_ablation_file(source_path, output_path)
            res,_,_,_ = process_mepo_batch(base_llm, source_path, evaluator, model=mepo_model, batch_size=batch_size)
            merge_res = merge_optim_prompt_with_original(
                res,
                output_path,
                MEPO
            )
            
torch.cuda.empty_cache()
gc.collect()

# =========================
# Step 2: Infer RMePO M candidate prompts for all items
# =========================
from collections import defaultdict

for base_llm in base_llm_models:
    for data_path in evaluation_datasets:
        for evaluator in evaluator_models:
            path = f'{folder_name}/{create_combined_name(base_llm, data_path, evaluator)}.json'
            print(f"Processing: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            all_prompts = []
            mapping = []

            for i, sample in enumerate(data):
                ori_prompt = sample.get("ori_prompt", "")
                for _ in range(M):
                    all_prompts.append(mepo_model.po_prompt_ins.replace("S_P", ori_prompt))
                    mapping.append(i)

            all_outputs = []
            for i in range(0, len(all_prompts), batch_size):
                batch = all_prompts[i:i+batch_size]
                outputs = mepo_model.generate_paraphrase_batch(batch)
                all_outputs.extend(outputs)

            grouped = defaultdict(list)

            assert len(mapping) == len(all_outputs)
            for idx, out in zip(mapping, all_outputs):
                grouped[idx].append(out)

            for i, sample in enumerate(data):
                sample["rmepo_paraphrases"] = grouped.get(i, [])

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
import torch, os, shutil, gc
from config import MODEL_CACHE_PATH
torch.cuda.empty_cache(), gc.collect()
    
if os.path.exists(MODEL_CACHE_PATH):
    shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)
