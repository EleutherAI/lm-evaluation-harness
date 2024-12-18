def verbalize_gender(gender):
    if gender == "Male":
        perspective = "un uomo"
    else:
        perspective = "una donna"
        
    return perspective

PROMPT_TEMPLATE = """
Sei {gender_verbalized}
Istruzione: Ti vengono fornite in input (Input) una coppia di frasi (Post, Reply) estratte da conversazioni sui social media. Il tuo compito è determinare se la Risposta (Reply) è ironica nel contesto del Post (Post). Fornisci in output (Output) una singola etichetta “ironia" o "non ironia".
Input:
Post: {post}
Reply: {reply}
Output: 
"""

def doc_to_text(x):
    return PROMPT_TEMPLATE.format(post=x["post"], reply=x["reply"], gender_verbalized=verbalize_gender(x["Gender"]))

