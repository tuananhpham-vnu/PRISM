import gc
import json
import os
import random
import shutil

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from config import MODEL_CACHE_PATH, SEED
from helper import (
    DEEPSEEK,
    DOLLY_EVAL,
    LLAMA2_7B,
    MINILM_EMBEDDING_MODEL,
    RMEPO,
    SELF_INSTRUCT_EVAL,
    M,
    create_combined_name,
    device,
    set_global_seed,
)

print("===== STEP 2: no-clustering multi-sampling ablation =====")
set_global_seed(SEED)
torch.cuda.empty_cache()
gc.collect()

method_keys = ["rmepo_paraphrases"]
folder_name = "ablation"
evaluation_datasets = [DOLLY_EVAL, SELF_INSTRUCT_EVAL]
evaluator_models = [DEEPSEEK]
METHOD = [RMEPO]
base_llm_models = [LLAMA2_7B]


def select_no_clustering_prompts(item, key, embedding_model, rng):
    samples = item.get(key, [])[:M]
    if not samples:
        return

    ori_prompt = item.get("ori_prompt", "")
    original_embedding = embedding_model.encode([ori_prompt], convert_to_tensor=True)
    candidate_embeddings = embedding_model.encode(samples, convert_to_tensor=True)
    sims_to_original = util.pytorch_cos_sim(original_embedding, candidate_embeddings)[0]

    random_idx = rng.randrange(len(samples))
    nearest_idx = torch.argmax(sims_to_original).item()
    farthest_idx = torch.argmin(sims_to_original).item()

    normalized_embeddings = F.normalize(candidate_embeddings, dim=1)
    centroid = F.normalize(normalized_embeddings.mean(dim=0, keepdim=True), dim=1)
    sims_to_centroid = util.pytorch_cos_sim(centroid, normalized_embeddings)[0]
    majority_idx = torch.argmax(sims_to_centroid).item()

    output_prefix = "rmepo"
    item[f"{output_prefix}_random_prompt"] = samples[random_idx]
    item[f"{output_prefix}_nearest_prompt"] = samples[nearest_idx]
    item[f"{output_prefix}_farthest_prompt"] = samples[farthest_idx]
    item[f"{output_prefix}_furthest_prompt"] = samples[farthest_idx]
    item[f"{output_prefix}_majority_prompt"] = samples[majority_idx]

    item[f"{output_prefix}_random_idx"] = random_idx
    item[f"{output_prefix}_nearest_idx"] = nearest_idx
    item[f"{output_prefix}_farthest_idx"] = farthest_idx
    item[f"{output_prefix}_furthest_idx"] = farthest_idx
    item[f"{output_prefix}_majority_idx"] = majority_idx
    item[f"{output_prefix}_similarity_to_original"] = [
        float(score) for score in sims_to_original
    ]
    item[f"{output_prefix}_similarity_to_candidate_centroid"] = [
        float(score) for score in sims_to_centroid
    ]


embed_model = SentenceTransformer(
    MINILM_EMBEDDING_MODEL,
    device=device,
    cache_folder=MODEL_CACHE_PATH,
)
rng = random.Random(SEED)

for base_llm in base_llm_models:
    for data_path in evaluation_datasets:
        for evaluator in evaluator_models:
            path = f"{folder_name}/{create_combined_name(base_llm, data_path, evaluator)}.json"
            print(f"Processing: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(len(data))
            for item in tqdm(data, desc=f"Selecting {os.path.basename(path)}"):
                for key in method_keys:
                    select_no_clustering_prompts(item, key, embed_model, rng)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved: {path}")

del embed_model
torch.cuda.empty_cache()
gc.collect()

if os.path.exists(MODEL_CACHE_PATH):
    shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)
