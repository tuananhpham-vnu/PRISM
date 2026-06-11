import gc
import json
import os

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
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
    distance_thresholds,
    set_global_seed,
)

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


def encode_candidates(item, embedding_model):
    samples = item.get(candidate_key, [])[:M]
    if not samples:
        return [], None, None

    ori_prompt = item.get("ori_prompt", "")
    original_embedding = embedding_model.encode([ori_prompt], convert_to_tensor=True)
    candidate_embeddings = embedding_model.encode(samples, convert_to_tensor=True)
    return samples, original_embedding, candidate_embeddings


def select_without_clustering(item, embedding_model):
    samples, original_embedding, candidate_embeddings = encode_candidates(item, embedding_model)
    if not samples:
        return

    sims_to_original = util.pytorch_cos_sim(original_embedding, candidate_embeddings)[0]
    best_idx = torch.argmax(sims_to_original).item()

    item[f"{output_prefix}_without_clustering_prompt"] = samples[best_idx]
    item[f"{output_prefix}_without_clustering_idx"] = best_idx
    item[f"{output_prefix}_without_clustering_similarity_to_original"] = [
        float(score) for score in sims_to_original
    ]


def cluster_candidate_indices(candidate_embeddings, distance_threshold):
    if len(candidate_embeddings) == 1:
        return [[0]]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(candidate_embeddings.cpu().numpy())

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    return list(clusters.values())


def select_cluster_median_representatives(samples, candidate_embeddings, clusters):
    representatives = []
    representative_scores = []

    normalized_embeddings = F.normalize(candidate_embeddings, dim=1)
    for cluster in clusters:
        if len(cluster) == 1:
            representatives.append(cluster[0])
            representative_scores.append(1.0)
            continue

        cluster_embeds = normalized_embeddings[cluster]
        median_embedding = torch.median(cluster_embeds, dim=0).values.unsqueeze(0)
        median_embedding = F.normalize(median_embedding, dim=1)
        sims_to_median = util.pytorch_cos_sim(median_embedding, cluster_embeds)[0]
        best_local_idx = torch.argmax(sims_to_median).item()

        representatives.append(cluster[best_local_idx])
        representative_scores.append(float(sims_to_median[best_local_idx]))

    return representatives, representative_scores


def select_without_consensus(item, embedding_model, distance_threshold):
    samples, _, candidate_embeddings = encode_candidates(item, embedding_model)
    if not samples:
        return

    clusters = cluster_candidate_indices(candidate_embeddings, distance_threshold)
    representatives, representative_scores = select_cluster_median_representatives(
        samples,
        candidate_embeddings,
        clusters,
    )

    largest_cluster_pos = max(
        range(len(clusters)),
        key=lambda idx: (len(clusters[idx]), -min(clusters[idx])),
    )
    best_idx = representatives[largest_cluster_pos]

    item[f"{output_prefix}_without_consensus_prompt"] = samples[best_idx]
    item[f"{output_prefix}_without_consensus_idx"] = best_idx
    item[f"{output_prefix}_without_consensus_cluster_id"] = largest_cluster_pos
    item[f"{output_prefix}_without_consensus_clusters"] = [
        [samples[idx] for idx in cluster] for cluster in clusters
    ]
    item[f"{output_prefix}_without_consensus_cluster_representatives"] = [
        samples[idx] for idx in representatives
    ]
    item[f"{output_prefix}_without_consensus_representative_scores"] = representative_scores


embed_model = SentenceTransformer(
    embedding_model_name,
    device=device,
    cache_folder=MODEL_CACHE_PATH,
)
distance_threshold = distance_thresholds.get(embedding_model_name)
assert distance_threshold is not None, f"Missing distance threshold for {embedding_model_name}"

for base_llm in base_llm_models:
    for data_path in evaluation_datasets:
        for evaluator in evaluator_models:
            path = f"{folder_name}/{create_combined_name(base_llm, data_path, evaluator)}.json"
            print(f"Processing: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in tqdm(data, desc=f"Selecting {os.path.basename(path)}"):
                select_without_clustering(item, embed_model)
                select_without_consensus(item, embed_model, distance_threshold)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved: {path}")

del embed_model
torch.cuda.empty_cache()
gc.collect()
