from lm_eval.utils import macro_f1_score, micro_f1_score, weighted_f1_score


def doc_to_text(doc):
    output = """You will be given a text snippet and 3 category definitions. 
    Your task is to choose which category applies to this text. 

    Your text snippet is: {tweet}

    Your category definitions are:

    HATE category definition: Hate speech is public speech that expresses hate or encourages violence towards a person 
    or group based on something such as race, religion, gender, ethnicity, sexual orientation or other characteristics.
    
    ABUSE category definition: Abusive and offensive language means verbal messages that use words in an inappropriate 
    way and may include but is not limited to swearing, name-calling, or profanity. Offensive language may upset or 
    embarrass people because it is rude or insulting
    
    NORMAL category definition: Normal language is neither hateful nor abusive or offensive. It does not contain any bad language.
    
    Does the text snippet belong to the HATE, ABUSIVE, or the NORMAL category?
    Thinking step by step answer HATE, ABUSIVE, or NORMAL capitalizing all the letters. 
    Explain your reasoning FIRST, then output HATE, ABUSIVE, or NORMAL.
  
    """

    text = output.format(tweet=doc["tweet"])
    return text

