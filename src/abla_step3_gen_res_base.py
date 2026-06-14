import gc
import json
import os
import shutil

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from config import MODEL_CACHE_PATH, prompt_template_vicuna
from helper import (
    DEEPSEEK,
    DOLLY_EVAL,
    LLAMA2_7B,
    SELF_INSTRUCT_EVAL,
    VICUNA_7B,
    base_llm_models,
    create_combined_name,
    device,
    evaluation_datasets,
    evaluator_models,
    load_model_and_tokenizer,
)
from utils import generate_batch


print("===== STEP 3.0: Base Response Generation =====")
torch.cuda.empty_cache()
gc.collect()

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

folder_name = "ablation"
evaluation_datasets = [DOLLY_EVAL, SELF_INSTRUCT_EVAL]
evaluator_models = [DEEPSEEK]
base_llm_models = [LLAMA2_7B]

PROMPT_RESPONSE_KEYS = [
    ("ori_prompt", "ori_response"),
    ("mepo_prompt", "mepo_response"),
]


def generate_base_responses(
    item,
    model,
    tokenizer,
    is_vicuna,
    is_need_context,
):
    available_pairs = [
        (prompt_key, response_key, item.get(prompt_key))
        for prompt_key, response_key in PROMPT_RESPONSE_KEYS
        if isinstance(item.get(prompt_key), str) and item[prompt_key].strip()
    ]
    if not available_pairs:
        return

    # Generate only once if ori_prompt and mepo_prompt are identical.
    unique_prompts = list(dict.fromkeys(prompt for _, _, prompt in available_pairs))
    prompts_for_generation = unique_prompts

    if is_vicuna:
        context = item.get("context") if is_need_context else None
        if isinstance(context, str) and context.strip():
            vicuna_prompts = [
                f"Context:\n{context}\n\nQuestion:\n{prompt}"
                for prompt in unique_prompts
            ]
        else:
            vicuna_prompts = unique_prompts

        prompts_for_generation = [
            prompt_template_vicuna.format(prompt) for prompt in vicuna_prompts
        ]

    unique_responses = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_for_generation,
        context=item.get("context") if is_need_context else None,
        do_sample=False,
        apply_chat_template=not is_vicuna,
        device=device,
    )
    prompt_to_response = dict(zip(unique_prompts, unique_responses))

    for _, response_key, prompt in available_pairs:
        item[response_key] = prompt_to_response[prompt]


for base_model in base_llm_models:
    available_files = []
    for data_path in evaluation_datasets:
        for evaluator in evaluator_models:
            file_name = create_combined_name(base_model, data_path, evaluator)
            input_path = os.path.join(folder_name, f"{file_name}.json")
            if os.path.exists(input_path):
                available_files.append((data_path, evaluator, input_path))

    if not available_files:
        print(f"Skipping {base_model}: no ablation input files found")
        continue

    torch.cuda.empty_cache()
    gc.collect()
    is_vicuna = base_model == VICUNA_7B
    model, tokenizer = load_model_and_tokenizer(
        model_path=base_model,
        cache_dir=MODEL_CACHE_PATH,
        token=hf_token,
    )

    for data_path, evaluator, input_path in available_files:
        is_need_context = data_path in [DOLLY_EVAL, SELF_INSTRUCT_EVAL]

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(
            f"\nBase model: {base_model} | Dataset: {data_path} | "
            f"Evaluator: {evaluator} | Loaded {len(data)} samples"
        )

        for item in tqdm(data, desc=f"Generating {os.path.basename(input_path)}"):
            generate_base_responses(
                item=item,
                model=model,
                tokenizer=tokenizer,
                is_vicuna=is_vicuna,
                is_need_context=is_need_context,
            )

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {input_path}")

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if os.path.exists(MODEL_CACHE_PATH):
    shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)
