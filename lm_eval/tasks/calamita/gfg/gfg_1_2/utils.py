import datasets
import itertools
import numpy as np

from evaluate import load


PROMPT_TEMPLATE = """Identifica le espressioni che contengono dei marcatori di genere femminile o maschile. Se più di un'espressione è identificata, separale con ; . Se nessuna espressione è identificata restituisci 0.

[Genere marcato]: Sempre basso è il numero dei laureati in corso (7).
[Espressione]: dei laureati

[Genere marcato]: Le pubblicazioni dei ricercatori DIMI nell’archivio IRIS-OPENBS.
[Espressione]: dei ricercatori

[Genere marcato]: A livello di Ateneo ha partecipato il 76% degli  iscritti del 1° anno CdL e l ’81% di tutti gli altri studenti.
[Espressione]: degli iscritti ; di tutti gli altri studenti

[Genere marcato]: {sentence}"""


def process_docs(dataset: datasets.Dataset):

    sentences = dataset['text']
    list_spans = []
    for entry in dataset:
        spans=[]
        for s in entry['list_spans']:
            spans.append(s['span'])
        if len(spans) < 1:
            list_spans.append(['0'])
        else:
            list_spans.append(spans)
    
    dataset_dict = {"sentence": sentences, 'list_spans': list_spans}
    return datasets.Dataset.from_dict(dataset_dict)

def doc_to_text(x):
    return PROMPT_TEMPLATE.format(sentence=x["sentence"])


def separate(model_output):
    separator = "]:"
    if separator not in model_output:
        return model_output
    else:
        index = model_output.rfind(separator)
        return model_output[index + len(separator):].strip()


def span_split(output):
    """
    Pre-processes the output line to be evaluated, splitting it into a list of identified spans.
    """
    if output is not None:
        spans = output.lower().strip()
        if ';' in spans:
            spans = spans.split(";")
        elif ',' in spans:
            spans = spans.split(",")
        else:
            spans = [spans]
    else:
        spans = ['0']
    return spans


BERT_SCORER = None


def bert_f1(predictions, references):
    
    global BERT_SCORER
    if BERT_SCORER is None:
        BERT_SCORER = load("bertscore", keep_in_memory=True)

    result = span_split(separate(predictions[0]))
    spans = references
        
    list_level_f1 = []
    # compute couples: generated output with the most corresponding annotation by means of bertscore
    couples = list(itertools.product(spans, result))
    scores = []
    for p in couples:
        if '0' not in p:
            score = BERT_SCORER.compute(predictions=[p[0]], references=[p[1]], lang="it")["f1"]
            scores.append(score)
        else:
            scores.append([0.0])
    scores = np.array(scores)
    unflatten_scores = scores.reshape(len(spans), -1).T # N_output X N_spans

    # reduce matrix to the most relevant couples
    for j in range(max(len(spans), len(result))):
        # Take the max 
        max_score_indices = np.unravel_index(unflatten_scores.argmax(), unflatten_scores.shape)
        max_ = unflatten_scores[max_score_indices[0],max_score_indices[1]]
        list_level_f1.append(max_)
        unflatten_scores[max_score_indices[0],:] = 0
        unflatten_scores[:,max_score_indices[1]] = 0

    return np.mean(list_level_f1)
