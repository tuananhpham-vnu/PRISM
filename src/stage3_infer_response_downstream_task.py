from helper import *

folder_name = downstream_folder_name

embedding_models = [
    GEMMA_EMBEDDING_MODEL,
    MINILM_EMBEDDING_MODEL
]

METHOD = [
    RBPO,
    RMEPO
]

for method in METHOD:
    for embed_model_name in embedding_models:
        for llm_model in base_llm_models:
            model, tokenizer = load_model_and_tokenizer(llm_model)
            for task in downstream_task_datasets:
                if task.upper () == 'BBH':
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
                    save_path = f"{downstream_folder_name}/{clean_name(embed_model_name)}/{method}/{task}/{subtask}_response.json"
                else:
                    save_path = f"{downstream_folder_name}/{clean_name(embed_model_name)}/{method}/{task}_response.json"