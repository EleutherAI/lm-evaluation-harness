import torch
import datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

PROMPT_TEMPLATE = """Riformula la seguente frase utilizzando un linguaggio neutro rispetto al genere dei referenti umani, evitando l’uso di forme maschile e femminili.

[Genere marcato]: Secondariamente, fino a che punto aumenta la trasparenza e la responsabilità dei parlamentari europei?
[Neutro]: Secondariamente, fino a che punto aumenta la trasparenza e la responsabilità dei membri del Parlamento Europeo?

[Genere marcato]: Signora Presidente, su tali questioni sarà necessario che tutti continuino a dare prova d'ambizione.
[Neutro]: Presidente, su tali questioni sarà necessario che ogni persona continui a dare prova d'ambizione.

[Genere marcato]: A Belgrado, molti pescatori si sono schierati dalla parte dei politici.
[Neutro]: A Belgrado, molte persone che lavorano nella pesca hanno preso le parti di chi fa politica.

[Genere marcato]: Non vogliamo certo maltrattare i produttori di mangimi composti e di alimenti per animali, né vogliamo maltrattare i poveri agricoltori.
[Neutro]: Non vogliamo certo maltrattare coloro che producono mangimi composti e alimenti per animali, né vogliamo maltrattare le povere persone che lavorano nell’agricoltura.

[Genere marcato]: {sentence}"""


def process_docs(dataset: datasets.Dataset):
    dataset = dataset.filter(lambda entry: entry['SET'] == 'Set-N')
    return dataset

def doc_to_text(x):
    return PROMPT_TEMPLATE.format(sentence=x["REF-G"])

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
     
    if 0 == new_label:
        return 1
    else:
        return 0

