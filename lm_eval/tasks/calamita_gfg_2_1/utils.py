import torch
import datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


PROMPT_TEMPLATE = """Riformula la seguente frase utilizzando un linguaggio neutro rispetto al genere dei referenti umani, evitando l’uso di forme maschile e femminili.

[Genere marcato]: - Il 79% dei laureati lavora (a un anno dalla laurea).
[Neutro]: - Il 79% delle persone laureate lavora (a un anno dalla laurea).

[Genere marcato]: È previsto, altresì, un rappresentante del Personale Tecnico-Amministrativo.
[Neutro]: È prevista, altresì, una persona in rappresentanza del Personale Tecnico-Amministrativo.

[Genere marcato]: Decreto Rettorale n. 235/2021 di nomina dei nuovi rappresentanti degli studenti
[Neutro]: Decreto Rettorale n. 235/2021 di nomina dei nuovi membri di rappresentanza della comunità studentesca

[Genere marcato]: {sentence}"""


def get_label(entry):
    columns = ["list_spans", "rewritten_texts_generico", "rewritten_texts_sovraesteso"]
    if len(entry[columns[0]]) == 0: #list_span empty
        return None
    else:
        if len(entry[columns[1]]) == 0 and len(entry[columns[2]]) == 0:
            return 1 #gendered
        else:
            return 0 #neutral


def process_docs(dataset: datasets.Dataset):
    dataset = dataset.map(lambda x: {"sentence": PROMPT_TEMPLATE.format(sentence=x["text"]), "label": get_label(x)})
    dataset = dataset.filter(lambda x: x["label"] is not None)
    return dataset


def separate(model_output):
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
