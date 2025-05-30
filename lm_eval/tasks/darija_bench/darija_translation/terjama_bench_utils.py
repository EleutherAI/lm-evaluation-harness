
def input_prompt_en_ary(dataset):
    prompt=f"English phrase: {dataset['English']} \nTranslate to Moroccan Arabic Darija (in Arabic script only):"
    return prompt

def target_output_en_ary(dataset):
    return dataset["Darija"]

def input_prompt_ary_en(dataset):
    prompt=f"Moroccan Arabic Darija phrase: {dataset['Darija']} \nTranslate to English:"
    return prompt

def target_output_ary_en(dataset):
    return dataset["English"]

