import re

def doc_to_text_debug(doc) -> str:
    word_guesses = "\n".join("- [Edificio religioso] = "+d for d in doc['word_guesses'].split(";"))
    sol_words = "\n".join("11 = "+w for w in doc["solution_words"].split(";"))
    PROMPT_TEMPLATE = f"""
    Rimanda il seguente testo una sola volta senza modifiche:
    {word_guesses}
    Prima lettura: {doc['first_pass']},
    {sol_words},
    Soluzione: {doc['solution']},
    """
    return PROMPT_TEMPLATE

def doc_to_text(doc) -> str:
    verbalized_rebus = doc["verbalized_rebus"]
    solution_key = doc["solution_key"]

    PROMPT_TEMPLATE = f"""Sei un'esperto risolutore di giochi enigmistici. Il seguente gioco contiene una frase (Rebus) nella quale alcune parole sono state sostituite da indizi tra parentesi quadre. Il tuo compito è quello di identificare le parole nascoste e sostituirle agli indizi nel Rebus, producendo una prima lettura dalla quale poi si deriverà una frase risolutiva. La chiave di lettura è una sequenza di numeri che rappresentano la rispettive lunghezze delle parole che compongono la frase risolutiva. La tua risposta deve essere una frase risolutiva sensata e che rispetti le lunghezze definite nella chiave di lettura.
# Esempio 1:
Rebus: AC [Un mollusco nell'insalata di mare] GLI [Lo è l'operaio che lavora in cantiere] S TO [Soldati da trincea]
Chiave di lettura: 11 2 10
Procediamo alla risoluzione del rebus passo per passo:
- A C = A C
- [Un mollusco nell'insalata di mare] = cozza
- G L I = G L I
- [Lo è l'operaio che lavora in cantiere] = edile
- S T O = S T O
- [Soldati da trincea] = fanti
Prima lettura: AC cozza GLI edile S TO fanti
Ora componiamo la soluzione seguendo la chiave risolutiva:
11 = Accozzaglie
2 = di
10 = lestofanti
Soluzione: Accozzaglie di lestofanti
# Esempio 2:
Rebus: [Edificio religioso] G [Lo fa doppio l'opportunista] NP [Poco cortese, severo] NZ [Parente... molto lontana]
Chiave di lettura: 3 1 6 3 8 2
Procediamo alla risoluzione del rebus passo per passo:
- [Edificio religioso] = chiesa
- G = G
- [Lo fa doppio l'opportunista] = gioco
- N P = N P
- [Poco cortese, severo] = rude
- N Z = N Z
- [Parente... molto lontana] = ava
Prima lettura: chiesa G gioco NP rude NZ ava
Ora componiamo la soluzione seguendo la chiave risolutiva:
3 = Chi
1 = è
6 = saggio
3 = con
8 = prudenza
2 = va
Soluzione: Chi è saggio con prudenza va
# Ora tocca a te!
Completa il rebus seguendo il procedimento descritto, rispondendo esattamente nello stesso formato utilizzato dagli esempi precedenti.
Rebus: {verbalized_rebus}
Chiave di lettura: {solution_key}
"""
    return PROMPT_TEMPLATE

def doc_to_text_hints(doc) -> str:
    verbalized_rebus = doc["verbalized_rebus_with_length_hints"]
    solution_key = doc["solution_key"]

    PROMPT_TEMPLATE = f"""Sei un'esperto risolutore di giochi enigmistici. Il seguente gioco contiene una frase (Rebus) nella quale alcune parole sono state sostituite da indizi tra parentesi quadre. I numeri in ogni indizio rappresentano la lunghezza della parola nascosta. Il tuo compito è quello di identificare le parole nascoste e sostituirle agli indizi nel Rebus, producendo una prima lettura dalla quale poi si deriverà una frase risolutiva. La chiave di lettura è una sequenza di numeri che rappresentano la rispettive lunghezze delle parole che compongono la frase risolutiva. La tua risposta deve essere una frase risolutiva sensata e che rispetti le lunghezze definite nella chiave di lettura.
# Esempio 1:
Rebus: AC [Un mollusco nell'insalata di mare (5)] GLI [Lo è l'operaio che lavora in cantiere (5)] S TO [Soldati da trincea (5)]
Chiave di lettura: 11 2 10
Procediamo alla risoluzione del rebus passo per passo:
- A C = A C
- [Un mollusco nell'insalata di mare] = cozza
- G L I = G L I
- [Lo è l'operaio che lavora in cantiere] = edile
- S T O = S T O
- [Soldati da trincea] = fanti
Prima lettura: AC cozza GLI edile S TO fanti
Ora componiamo la soluzione seguendo la chiave risolutiva:
11 = Accozzaglie
2 = di
10 = lestofanti
Soluzione: Accozzaglie di lestofanti
# Esempio 2:
Rebus: [Edificio religioso (6)] G [Lo fa doppio l'opportunista (5)] NP [Poco cortese, severo (4)] NZ [Parente... molto lontana (3)]
Chiave di lettura: 3 1 6 3 8 2
Procediamo alla risoluzione del rebus passo per passo:
- [Edificio religioso] = chiesa
- G = G
- [Lo fa doppio l'opportunista] = gioco
- N P = N P
- [Poco cortese, severo] = rude
- N Z = N Z
- [Parente... molto lontana] = ava
Prima lettura: chiesa G gioco NP rude NZ ava
Ora componiamo la soluzione seguendo la chiave risolutiva:
3 = Chi
1 = è
6 = saggio
3 = con
8 = prudenza
2 = va
Soluzione: Chi è saggio con prudenza va
# Ora tocca a te!
Completa il rebus seguendo il procedimento descritto, rispondendo esattamente nello stesso formato utilizzato dagli esempi precedenti.
Rebus: {verbalized_rebus}
Chiave di lettura: {solution_key}
"""
    return PROMPT_TEMPLATE


def doc_to_target(doc):
  return  ""

def process_results(doc, results):
  # doc: original doc
  # results[0]: string output of the model
  try:
    model_generation = results[0]
    regex_word_guesses = r'[\d\.|-] \[.* = (.*)'
    regex_firstpass = r'Prima lettura: (.*)'
    regex_solution_word = r"\d+ = (.*)"
    regex_solution = r"Soluzione: (.*)"
  
    try:
        word_guesses = ";".join(re.findall(regex_word_guesses, model_generation))
    except:
        word_guesses = ""

    try:
        first_pass = re.findall(regex_firstpass, model_generation)[0]
    except:
        first_pass = ""

    try:
        solution_words = ";".join(re.findall(regex_solution_word, model_generation))
    except:
        solution_words = ""

    try:
        solution = re.findall(regex_solution, model_generation)[0]
    except:
        solution = ""

    parsed_generation = {
        "word_guesses": word_guesses,
        "first_pass": first_pass,
        "solution_words": solution_words,
        "solution": solution,
    }
    gold_fields= {
        "word_guesses": doc['word_guesses'],
        "first_pass": doc['first_pass'],
        "solution_words": doc['solution_words'],
        "solution": doc['solution'],
    }
    """Returns a dictionary of metrics evaluating the step-by-step resolution of a set of verbalized rebus.
    Args:
        parsed_generation (dict[str, str]): A list of outputs, each produced by parse_generation_elements.
        gold_fields (dict[str, str]): List of dictionaries containing gold values matching for word_guesses, first_pass,
            solution_words and solution, taken from test.csv
    Returns:
        dict[str, float]: A dictionary containing granular scores representing LLM performances on various stages of
            rebus solving for a set of rebuses (see https://arxiv.org/abs/2408.00584 for reference)
    Example inputs:
        parsed_generation = [{
            "word_guesses": Adamo;Eva;more,
            "first_pass": Adamo R Eva D A A more,
            "solution_words": Ad;amore;vada;amore,
            "solution": Ad amore vada amore,
        }]
        gold_fields = [{
            "word_guesses": Eva;Eva;more,
            "first_pass": Eva R Eva D A A more,
            "solution_words": Ev;areva;daam;ore,
            "solution": Ev areva daam ore,
        }]
    """
    
    ## DEBUG to test with faulty model generation
    #if len(parsed_generation["word_guesses"]) == 0:
    #  parsed_generation["word_guesses"] = gold_fields["word_guesses"][:-2] + "kw"
    #if len(parsed_generation["first_pass"]) == 0:
    #  parsed_generation["first_pass"] = " ".join(gold_fields["first_pass"].split(" ")[:-2]) + " unic P"
    ###

    def get_flat_wordlists(field: str, gold: dict[str, str], pred: dict[str, str]) -> float:
        gold_flat = list(gold[field].split(";") if isinstance(gold[field], str) else [])
        pred_flat = list(pred[field].split(";") if isinstance(pred[field], str) else [])
        #assert len(gold_flat) == len(pred_flat[:len(gold_flat)])
        return gold_flat, pred_flat[:len(gold_flat)]

    def get_words_accuracy(field: str, gold: dict[str, str], pred: dict[str, str]) -> float:
        gold_flat, pred_flat = get_flat_wordlists(field, gold, pred)
        matches = [int(g.lower() == p.lower()) for g, p in zip(gold_flat, pred_flat)]
        return round(sum(matches) / len(matches), 4)

    def get_fulltext_accuracy(field: str, gold: dict[str, str], pred: dict[str, str]) -> float:
        return int(pred[field].lower() == gold[field].lower()) 

    gold_solution_words_flat, pred_solution_words_flat = get_flat_wordlists("solution_words", gold_fields, parsed_generation)
    solution_length_matches = [int(len(g) == len(p)) for g, p in zip(gold_solution_words_flat, pred_solution_words_flat)]
    
    try:
      word_guesses_accuracy = get_words_accuracy("word_guesses", gold_fields, parsed_generation)
    except:
      word_guesses_accuracy = 0
    try:
      first_pass_accuracy= get_fulltext_accuracy("first_pass", gold_fields, parsed_generation)
    except:
      first_pass_accuracy = 0
    try:
      solution_words_accuracy= get_words_accuracy("solution_words", gold_fields, parsed_generation)
    except:
      solution_words_accuracy = 0
    try:
      solution_words_lengths_accuracy= round(sum(solution_length_matches) / len(solution_length_matches), 4)
    except:
      solution_words_lengths_accuracy = 0
    try:
      solution_match= get_fulltext_accuracy("solution", gold_fields, parsed_generation)
    except:
      solution_match = 0
  except:
    raise Exception("metrics error")
  return {"word_guesses_accuracy": word_guesses_accuracy, "first_pass_accuracy": first_pass_accuracy, "solution_words_accuracy": solution_words_accuracy, "solution_words_lengths_accuracy": solution_words_lengths_accuracy, "solution_match": solution_match}

#  "word_guesses_accuracy",
#    "first_pass_accuracy",
#    "solution_words_accuracy",
#    "solution_words_lengths_accuracy",
#    "solution_match"

def preprocess_dataset(dataset):
    dataset = dataset.select([i for i in range(4)])      # selecting 4 rows for DEBUG
    return dataset

###### original functions from sarti
def parse_generation_elements(doc, results) -> dict[str, str]:
    """Extracts reasoning steps from the generated model answer. If the model does not comply with the requirement of
    matching in-context examples format, the corresponding malformed parts are set to empty strings
    Args:
        model_generation (str): The answer generated by the LLM.
    Returns:
        dict[str, str]: A dictionary containing the following fields:
        - word_guesses: Solutions for the crossword definitions in the verbalized rebus, separated by ;
        - first_pass: The first pass produced by replacing definitions by word guesses in the verbalized rebus
        - solution_words: Words composing the solution generated by the model, separated by ;
        - solution: The full solution generated by the model
    """
    import re
    regex_word_guesses = r'[\d\.|-] \[.* = (.*)'
    regex_firstpass = r'Prima lettura: (.*)'
    regex_solution_word = r"\d+ = (.*)"
    regex_solution = r"Soluzione: (.*)"

    model_generation = results[0]
  
    try:
        word_guesses = ";".join(re.findall(regex_word_guesses, model_generation))
    except:
        word_guesses = ""

    try:
        first_pass = re.findall(regex_firstpass, model_generation)[0]
    except:
        first_pass = ""

    try:
        solution_words = ";".join(re.findall(regex_solution_word, model_generation))
    except:
        solution_words = ""

    try:
        solution = re.findall(regex_solution, model_generation)[0]
    except:
        solution = ""

    parsed_generation = [{
        "word_guesses": word_guesses,
        "first_pass": first_pass,
        "solution_words": solution_words,
        "solution": solution,
    }]

    # (optional). The post-processing function to be applied to every model output.
    POSTPROCESSING_FUNC = parse_generation_elements

    # NOTE: The main metric is "solution_match"
    METRIC_LIST = [
        "word_guesses_accuracy",
        "first_pass_accuracy",
        "solution_words_accuracy",
        "solution_words_lengths_accuracy",
        "solution_match"
    ]

    
    gold_fields= [{
        "word_guesses": doc['word_guesses'],
        "first_pass": doc['first_pass'],
        "solution_words": doc['solution_words'],
        "solution": doc['solution'],
    }]

    """Returns a dictionary of metrics evaluating the step-by-step resolution of a set of verbalized rebus.
    Args:
        parsed_generation (dict[str, str]): A list of outputs, each produced by parse_generation_elements.
        gold_fields (dict[str, str]): List of dictionaries containing gold values matching for word_guesses, first_pass,
            solution_words and solution, taken from test.csv
    Returns:
        dict[str, float]: A dictionary containing granular scores representing LLM performances on various stages of
            rebus solving for a set of rebuses (see https://arxiv.org/abs/2408.00584 for reference)
    Example inputs:
        parsed_generation = [{
            "word_guesses": Adamo;Eva;more,
            "first_pass": Adamo R Eva D A A more,
            "solution_words": Ad;amore;vada;amore,
            "solution": Ad amore vada amore,
        }]
        gold_fields = [{
            "word_guesses": Eva;Eva;more,
            "first_pass": Eva R Eva D A A more,
            "solution_words": Ev;areva;daam;ore,
            "solution": Ev areva daam ore,
        }]
    """
    def get_value_or_empty(list: list, item_idx: int):
        if len(list) > item_idx:
            return list[item_idx]
        return ""

    def get_flat_wordlists(field: str, gold: list[dict[str, str]], pred: list[dict[str, str]]) -> float:
        gold_nested = list(d[field].split(";") if isinstance(d[field], str) else [] for d in gold)
        pred_nested = list(d[field].split(";") if isinstance(d[field], str) else [] for d in pred)
        gold_flat = [item for sublist in gold_nested for item in sublist]
        # Flatten and ensure to keep entries aligned with gold if the number of words is mismatching
        pred_flat = [
            get_value_or_empty(pred_nested[entry_idx], item_idx) for entry_idx, entry in enumerate(gold_nested)
            for item_idx, _ in enumerate(entry)
        ]
        assert len(gold_flat) == len(pred_flat)
        return gold_flat, pred_flat

    def get_words_accuracy(field: str, gold: list[dict[str, str]], pred: list[dict[str, str]]) -> float:
        gold_flat, pred_flat = get_flat_wordlists(field, gold, pred)
        matches = [int(g.lower() == p.lower()) for g, p in zip(gold_flat, pred_flat)]
        return round(sum(matches) / len(matches), 4)

    def get_fulltext_accuracy(field: str, gold: list[dict[str, str]], pred: list[dict[str, str]]) -> float:
        matches = [
            int(pred_dic[field].lower() == gold_dic[field].lower()) 
            for pred_dic, gold_dic in zip(pred, gold)
        ]
        return round(sum(matches) / len(matches))

    gold_solution_words_flat, pred_solution_words_flat = get_flat_wordlists("solution_words", gold_fields, parsed_generation)
    solution_length_matches = [int(len(g) == len(p)) for g, p in zip(gold_solution_words_flat, pred_solution_words_flat)]
    return {
        "word_guesses_accuracy": get_words_accuracy("word_guesses", gold_fields, parsed_generation),
        "first_pass_accuracy": get_fulltext_accuracy("first_pass", gold_fields, parsed_generation),
        "solution_words_accuracy": get_words_accuracy("solution_words", gold_fields, parsed_generation),
        "solution_words_lengths_accuracy": round(sum(solution_length_matches) / len(solution_length_matches), 4),
        "solution_match": get_fulltext_accuracy("solution", gold_fields, parsed_generation)
    }