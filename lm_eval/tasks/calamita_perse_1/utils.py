def verbalize_age(generation):
    if generation == "GenZ":
        perspective = "una persona giovane della generazione Z"
    elif generation == "GenY":
        perspective = "una persona giovane della generazione Y"
    elif generation == "GenX":
        perspective = "una persona adulta della generazione X"
    else:
        perspective = "una persona adulta della generazione baby boomer"
        
    return perspective

PROMPT_TEMPLATE = """
Sei {age_verbalized}
Istruzione: Ti vengono fornite in input (Input) una coppia di frasi (Post, Reply) estratte da conversazioni sui social media. Il tuo compito è determinare se la Risposta (Reply) è ironica nel contesto del Post (Post). Fornisci in output (Output) una singola etichetta “ironia" o "non ironia".
Input:
Post: {post}
Reply: {reply}
Output: 
"""

def doc_to_text(x):
    return PROMPT_TEMPLATE.format(post=x["post"], reply=x["reply"], age_verbalized=verbalize_age(x['Generation']))

