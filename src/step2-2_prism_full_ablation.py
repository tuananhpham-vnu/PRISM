import gc
import os
import shutil
from sentence_transformers import util
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import torch
from config import MODEL_CACHE_PATH, SEED
from helper import DEEPSEEK, DOLLY_EVAL, LLAMA2_7B, RMEPO, SELF_INSTRUCT_EVAL, clean_name, eval_folder_name, device, M, distance_thresholds, METHOD, MINILM_EMBEDDING_MODEL, set_global_seed, create_combined_name
from step2_clustering_and_selecting import prompt_clustering, representative_selection, compute_consensus_score, optimize_prompt_selection

print("===== STEP 2.1: PRISM ablation selection =====")
set_global_seed(SEED)
torch.cuda.empty_cache()
gc.collect()

folder_name = "ablation"
candidate_key = "rmepo_paraphrases"
output_prefix = "rmepo"
evaluation_datasets = [DOLLY_EVAL, SELF_INSTRUCT_EVAL]
evaluator_models = [DEEPSEEK]
METHOD = [RMEPO]
base_llm_models = [LLAMA2_7B]
embedding_model_name = MINILM_EMBEDDING_MODEL

embed_model = SentenceTransformer(
    embedding_model_name,
    device=device,
    cache_folder=MODEL_CACHE_PATH
)

distance_threshold = distance_thresholds.get(embedding_model_name,None)
print(f"Using embedding model: {embedding_model_name} with distance threshold: {distance_threshold}")

assert distance_threshold is not None, f"Distance threshold not found for embedding model '{embedding_model_name}'"

for base_llm in base_llm_models:
    for data_path in evaluation_datasets:
        for evaluator in evaluator_models:
            path = f"{folder_name}/{create_combined_name(base_llm, data_path, evaluator)}.json"
            print(f"Processing: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(len(data))
            
            for item in data:
                clusters, embeddings = prompt_clustering(candidate_key, item, embed_model, M, distance_threshold)
                if clusters is None or len(clusters) == 0:  # Bỏ qua nếu dữ liệu không đủ
                    print(f"Warning: Không thể thực hiện clustering cho key '{candidate_key}' do dữ liệu không đủ, bỏ qua item này")
                    continue
                cluster_representatives, single_cluster = representative_selection(item, embed_model,clusters, embeddings)
                consensus_scores = compute_consensus_score(candidate_key, item, embed_model, clusters, embeddings, cluster_representatives, single_cluster)
                best_rep_idx = optimize_prompt_selection(candidate_key, item, clusters, embeddings, cluster_representatives, consensus_scores)
                
                output_key = "rbpo" if candidate_key == "rbpo_paraphrases" else "rmepo"
                item[f'{output_key}_clusters'] = [[item[candidate_key][idx] for idx in cluster] for cluster in clusters]
                item[f'{output_key}_prompt'] = item[candidate_key][best_rep_idx]
                item[f"{output_key}_cluster_representatives"] = [item[candidate_key][idx] for idx in cluster_representatives]
                item[f"{output_key}_consensus_scores"] = consensus_scores
                
                for key in METHOD:
                    key_to_remove = f"{key}_response"
                    if item.get(key_to_remove) is not None:
                        del item[key_to_remove]
            output_path = f'{eval_folder_name}/{clean_name(embedding_model_name)}/{path}.json'
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
# remove MODEL_CACHE_PATH
if os.path.exists(MODEL_CACHE_PATH):
    shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)