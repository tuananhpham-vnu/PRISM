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
    MATH: [GSM8K,"multistep_arithmetic_two", "object_counting"], # NOTE: math template
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

multiple_choice = [
    'date_understanding', 
    'disambiguation_qa', 
    'hyperbaton', 
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects', 
    'logical_deduction_three_objects',
    'movie_recommendation', 
    'penguins_in_a_table', 
    'reasoning_about_colored_objects',
    'ruin_names', 
    'salient_translation_error_detection', 
    'snarks',
    'temporal_sequences', 
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects', 
    'tracking_shuffled_objects_three_objects',
    'causal_judgement',
    'formal_fallacies',
    'navigate',
    'web_of_lies',
    'sports_understanding',
    'boolean_expressions',
    'multistep_arithmetic_two', 
    'object_counting', 
    'word_sorting'
]

def check_main_task(task_name: str):
    for main_task, sub_tasks in template_type.items():
        if task_name in sub_tasks:
            return main_task
    return None

def make_prompt_template(
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
        
        if "A:" in prompt and "B:" in prompt:
            sol1 = prompt.split("A:")[1].split("B:")[0].strip()
            sol2 = prompt.split("B:")[1].strip()
        return prompt_template_multiple_choice.format(Q=prompt, sol1=sol1, sol2=sol2)
    else:
        return prompt
    


# input = """
# How would a typical person answer each of the following questions about causation?\nJanet is an employee in the factory. She works in the maintenance department where she monitors the stability of all machines. Since she works in the maintenance department, she knows how to grease and oil all of the machines in the factory. It is her responsibility to put oil into the machines. Kate is also an employee in the factory. She works in the human resources department where she monitors the salaries of all employees. While she works in the human resources department, she knows how to grease and oil all of the machines in the factory. If Janet does not put oil in the machines, it is not Kate's responsibility to do so. On June 1st, Janet forgot to put oil into the machine. The machine broke down. Did the machine break down because Kate did not put oil in the machine?\nOptions:\n- Yes\n- No
# """

# print(prompt_template_bbh_yes_no.format(Q=input))
print(check_main_task("date_understanding"))
print(check_main_task("causal_judgement"))
print(check_main_task(ARC_C))