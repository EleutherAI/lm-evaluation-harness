import re
import datasets
import itertools
import numpy as np

from evaluate import load


PROMPT_TEMPLATE = """Identifica le espressioni che contengono dei marcatori di genere femminile o maschile. Se più di un'espressione è identificata, separale con ; . Se nessuna espressione è identificata restituisci 0.

[Genere marcato]: Quest’anno mi sono ammalata già due volte.
[Espressione]: ammalata

[Genere marcato]: La studentessa era preoccupata di andare fuori tema.
[Espressione]: La studentessa ; preoccupata

[Genere marcato]: Le altre giocatrici al tavolo da poker sono le tue avversarie.
[Espressione]: Le altre giocatrici ; le tue avversarie

[Genere marcato]: {sentence}"""


def process_docs(dataset: datasets.Dataset):

    neogate_m = dataset['REF-M'][:420]
    neogate_f = dataset['REF-F'][420:]

    refs = dataset["REF-TAGGED"]

    sentences = neogate_m + neogate_f

    list_spans = []
    sentences_tags = []
    for ref in refs:
        ref_words = ref.split(' ')
        ref_tags = []
        for i, word in enumerate(ref_words):
            if '<' in word:
                ref_tags.append(i)
        sentences_tags.append(ref_tags)
    
    sentence_level_spans = []
    for tags in sentences_tags:
        if not tags:
            sentence_level_spans.append([])
            continue

        spans = []
        current_span = [tags[0]]
        for i in range(0, len(tags)):
            if tags[i] == tags[i - 1] + 1:
                current_span.append(tags[i])
            else:
                if len(current_span) > 1:
                    if current_span not in spans:
                        spans.append(current_span)
                current_span = [tags[i]]
                spans.append(current_span)                            
        sentence_level_spans.append(spans)

    for sentence, indexes in zip(sentences, sentence_level_spans):
        sentence = re.sub('[\']', ' ', sentence)
        span_sent = []
        for list_i in indexes:
            spans =[]
            for i in list_i:   
                token = sentence.split(' ')[i]
                token = re.sub('[,.:;!?]', ' ', token).strip()
                spans.append(token)
            if len(spans) > 1:
                span_sent.append(' '.join(spans))
            else:
                span_sent.append(spans[0])
        list_spans.append(span_sent)
    
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
