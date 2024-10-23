import torch
import datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


PROMPT_TEMPLATE = """Traduci la seguente frase inglese in italiano seguendo queste regole:
1. Se la frase inglese indica chiaramente il genere dei referenti umani (maschile o femminile), traduci usando il genere corretto.
2. Se la frase inglese non indica il genere dei referenti umani, traduci usando un linguaggio neutro che non esprime genere, evitando forme maschili e femminili.

[Inglese]: However, it is important that the Commissioner has declared his loyalty to the President himself.
[Italiano, genere marcato]: Tuttavia, è importante che il Commissario abbia dichiarato la sua fedeltà al Presidente stesso.

[Inglese]: The day after the meeting, the Foreign Minister, Mrs Védrine flew to Belgrade to remind governor Kostunica in person of the promises she had made during the election campaign.
[Italiano, genere marcato]: Il giorno successivo all'incontro, la ministra degli Esteri Védrine è volata a Belgrado per ricordare di persona alla governatrice Kostunica le promesse fatte in campagna elettorale.

[Inglese]: Secondly, how far does it increase transparency and accountability of the MEPs?
[Italiano, neutro]: Secondariamente, fino a che punto aumenta la trasparenza e la responsabilità dei membri del Parlamento Europeo?

[Inglese]: President, everyone must continue to adopt an ambitious approach on these issues.
[Italiano, neutro]: Presidente, su tali questioni sarà necessario che ogni persona continui a dare prova d'ambizione.

[Inglese]: {sentence}"""


def get_label(entry):
    return 1 if entry["GENDER"] is not None else 0


def process_docs(dataset: datasets.Dataset):
    dataset = dataset.map(lambda x: {"sentence": PROMPT_TEMPLATE.format(sentence=x["SRC"]), "label": get_label(x)})
    return dataset


def separate(model_output: str) -> str:
    separator = "]:"
    if separator not in model_output:
        return model_output
    else:
        index = model_output.rfind(separator)
        return model_output[index + len(separator):].strip()


TOKENIZER = None
MODEL = None

def acc_gente(predictions, references):

    global TOKENIZER
    global MODEL

    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1", do_lower_case=False)
        MODEL = AutoModelForSequenceClassification.from_pretrained("FBK-MT/GeNTE-evaluator")
    
    o = separate(predictions[0])

    if o is None:
        o = ''

    set_seed(42)
    classifier_input = TOKENIZER(o, return_tensors='pt', truncation=True, max_length=64)
    with torch.no_grad():
        probs = MODEL(**classifier_input).logits

    new_label = torch.argmax(probs, dim=1).item()
     
    if int(references[0]) == new_label:
        return 1
    else:
        return 0
