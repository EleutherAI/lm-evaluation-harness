def verbalize_age_gender(generation, gender):
    if generation == "GenZ" and gender == "male":
        perspective = "un giovane uomo della generazione Z"
    elif generation == "GenZ" and gender == "female":
        perspective = "una giovane donna della generazione Z"
        
    elif generation == "GenY" and gender == "male":
        perspective = "un giovane uomo della generazione Y"
    elif generation == "GenY" and gender == "female":
        perspective = "una giovane donna della generazione Y"
        
    elif generation == "GenX" and gender == "male":
        perspective = "un uomo adulto della generazione X"
    elif generation == "GenX" and gender == "female":
        perspective = "una donna adulta della generazione X"
        
    elif generation == "Boomer" and gender == "male":
        perspective = "un uomo adulto della generazione baby boomer"
    else:
      perspective = "una donna adulta della generazione baby boomer"
        
    return perspective

PROMPT_TEMPLATE = """
Sei {age_gender_verbalized}
Istruzione: Ti vengono fornite in input (Input) una coppia di frasi (Post, Reply) estratte da conversazioni sui social media. Il tuo compito è determinare se la Risposta (Reply) è ironica nel contesto del Post (Post). Fornisci in output (Output) una singola etichetta “ironia" o "non ironia".
Input:
Post: {post}
Reply: {reply}
Output: 
"""

def doc_to_text(x):
    return PROMPT_TEMPLATE.format(post=x["post"], reply=x["reply"], age_gender_verbalized=verbalize_age_gender(x["Generation"], x["Gender"]))
