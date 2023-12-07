import os
import spacy
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckNLI, SelfCheckBERTScore, SelfCheckNgram

# pip install spacy
# pip install selfcheckgpt
# python -m spacy download en

selfcheckgpt_type = os.environ.get('SELFCHECKGPTTYPE', 'SelfCheckNgram')
selfcheckgpt_device = os.environ.get('SELFCHECKGPTDEVICE', 'cpu')
selfcheckgpt_nlp = spacy.load("en_core_web_sm")

if selfcheckgpt_type == 'SelfCheckNgram':
    selfcheckgpt = SelfCheckNgram(n=1)
elif selfcheckgpt_type == 'SelfCheckBERTScore':
    selfcheckgpt = SelfCheckBERTScore(rescale_with_baseline=True)
elif selfcheckgpt_type == 'SelfCheckMQAG':
    selfcheckgpt = SelfCheckMQAG(device=selfcheckgpt_device)
elif selfcheckgpt_type == 'SelfCheckNLI':
    selfcheckgpt = SelfCheckNLI(device=selfcheckgpt_device)
else:
    raise ValueError(f"Wrong SELFCHECKGPTTYPE environment variable: {selfcheckgpt_type}")

print("Load selfcheckgpt successfully")


def doc_to_text(doc):
    doc_text = doc["wiki_bio_text"]
    doc_text = doc_text.split()
    doc_text = " ".join(doc_text[:5])
    doc_text = f"Please generating a Wikipedia passage starting with: {doc_text}\n"
    return doc_text


def doc_to_target(doc):
    answer = doc['wiki_bio_text']
    return answer

def process_results(doc, results, threshold=0.6):

    response_temperature_0 = results[0]
    other_responses = results[1:]
    passage = doc_to_target(doc)

    sentences = selfcheckgpt_nlp(response_temperature_0)
    sentences = [sent.text.strip() for sent in sentences.sents]
    if selfcheckgpt_type == 'SelfCheckNgram':
        selfcheckgpt_scores = selfcheckgpt.predict(
            sentences = sentences,
            passage = passage,
            sampled_passages = other_responses,
            )
    elif selfcheckgpt_type == 'SelfCheckBERTScore':
        selfcheckgpt_scores = selfcheckgpt.predict(
            sentences = sentences,
            sampled_passages = other_responses,
            )
    elif selfcheckgpt_type == 'SelfCheckMQAG':
        selfcheckgpt_scores = selfcheckgpt.predict(
            sentences = sentences,
            sampled_passages = other_responses,
            )
    elif selfcheckgpt_type == 'SelfCheckNLI':
        selfcheckgpt_scores = selfcheckgpt.predict(
            sentences = sentences,
            passage = passage,
            sampled_passages = other_responses,
            num_questions_per_sent = 5,          # number of questions to be drawn
            scoring_method = 'bayes_with_alpha', # options = 'counting', 'bayes', 'bayes_with_alpha'
            beta1 = 0.8, beta2 = 0.8,            # additional params depending on scoring_method
            )

    selfcheckgpt_scores_avg = sum(selfcheckgpt_scores) / len(selfcheckgpt_scores) if len(selfcheckgpt_scores) > 0 else 0
    selfcheckgpt_scores_max = max(selfcheckgpt_scores)

    return {'avg': selfcheckgpt_scores_avg, 'max': selfcheckgpt_scores_max}
