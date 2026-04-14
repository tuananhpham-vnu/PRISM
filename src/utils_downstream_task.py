from config import prompt_template_gsm8k, prompt_template_piqa, prompt_template_multiple_choice, prompt_template_bbh_yes_no,prompt_template_bbh_true_false, prompt_template_bbh_valid
from helper import GSM8K, BBH, ARC_C, ARC_E, PIQA


NO_TEMPLATE = "no_template"
PIQA = "piqa"
MATH = "math"
YES_NO = "yes_no"
TRUE_FALSE = "true_false"
VALID = "valid"
MULTIPLE_CHOICE = "multiple_choice"

template_type = {
    NO_TEMPLATE: ["word_sorting"], # NOTE: No template for this subtask
    PIQA: [PIQA], # NOTE: piqa template
    VALID: ['formal_fallacies'], # NOTE: select between "valid" and "invalid"
    TRUE_FALSE: ["boolean_expressions"], # NOTE: select between "True" and "False"
    MATH: [
        GSM8K,
        "multistep_arithmetic_two", 
        "object_counting"
    ], # NOTE: math template
    YES_NO: [
        "causal_judgement",
        'navigate',
        'web_of_lies', 
        'sports_understanding'
    ], # NOTE: select between "Yes" and "No"
    MULTIPLE_CHOICE: [
        ARC_C,
        ARC_E,
        'snarks',
        'ruin_names', 
        'hyperbaton', 
        'movie_recommendation',
        'penguins_in_a_table',
        'temporal_sequences',       
        'date_understanding',
        'disambiguation_qa', 
        'logical_deduction_three_objects',
        'logical_deduction_five_objects',
        'logical_deduction_seven_objects',
        'reasoning_about_colored_objects',
        'salient_translation_error_detection',
        'tracking_shuffled_objects_five_objects', 
        'tracking_shuffled_objects_seven_objects',
        'tracking_shuffled_objects_three_objects'
    ] # NOTE: select between Multiple choices, ARC
}

def check_main_task(task_name: str):
    for main_task, sub_tasks in template_type.items():
        if task_name in sub_tasks:
            return main_task
    return None

def format_prompt_template(
    task_name: str,
    item: dict,
    method_key: str,
):    
    assert item.get(f'{method_key}_prompt') is not None 
    prompt = item.get(f'{method_key}_prompt')
    
    task = check_main_task(task_name)
    if task == MATH:
        return prompt_template_gsm8k.format(Q=prompt)
    
    elif task == NO_TEMPLATE:
        return prompt
    
    elif task == YES_NO:
        return prompt_template_bbh_yes_no.format(Q=prompt)
    
    elif task == TRUE_FALSE:
        return prompt_template_bbh_true_false.format(Q=prompt)
    
    elif task == VALID:
        return prompt_template_bbh_valid.format(Q=prompt)

    elif task == PIQA:
        sol1 = item.get("sol1")
        sol2 = item.get("sol2")
        return prompt_template_piqa.format(Q=prompt, sol1=sol1, sol2=sol2)    
    
    elif task == MULTIPLE_CHOICE:
        def build_options_block(labels, texts):
            return "\n".join([f"##{l}: {t}" for l, t in zip(labels, texts)])
        def build_label_instruction(labels):
            return " or ".join([f"({l})" for l in labels])
        
        choices = item.get("choices")
        labels = choices.get("label")
        texts = choices.get("text")
        options_block = build_options_block(labels, texts)
        label_instruction = build_label_instruction(labels)
        return prompt_template_multiple_choice.format(Q=prompt,
            options_block=options_block,
            label_instruction=label_instruction
        )
        
    return prompt