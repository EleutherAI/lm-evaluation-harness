import evaluate


## This prompt is taken literally from the Jawaher paper, with a small adjustment to only output the explanation in Arabic.
PROMPT = """You are a language expert with deep knowledge or Arabic 'proverbs', cultural history, and their litrary meanings.
Your goal is to analyze the Arabic proverb and provide the following output:
An explanation of the proverb in Arabic.
Include any relevant background stories or cultural context that could be helpful for explanations.
Don't produce any other content than the request.
Proverb: {proverb}
"""


def doc_to_text(doc):
    cur_proverb = doc["Proverbs"]
    return PROMPT.format(proverb=cur_proverb)

# #Native
# def process_docs(domain):
#     def filter_fn(dataset):
#         return dataset.filter(lambda doc: doc.get("Variety") == domain and doc.get("Ar_Explanation"))
#     return filter_fn


# dialects = ['ALG',
#  'SAU',
#  'MOR',
#  'OMA',
#  'EGY',
#  'BAH',
#  'JOR',
#  'LIB',
#  'MAU',
#  'QAT',
#  'IRQ',
#  'PAL',
#  'LEB',
#  'TUN',
#  'MSA',
#  'UAE',
#  'SUD',
#  'SYR',
#  'KUW',
#  'YEM']

# for dia in dialects:
#     globals()[f'process_docs_{dia}'] = process_docs(dia)

def process_docs(domain):
    def filter_fn(dataset):
        return dataset.filter(lambda doc: doc.get("Variety") == domain and doc.get("Ar_Explanation"))
    return filter_fn

# process_docs_ALG = process_docs("ALG")
# process_docs_SAU = process_docs("SAU")
# process_docs_MOR = process_docs("MOR")
# process_docs_OMA = process_docs("OMA")
# process_docs_EGY = process_docs("EGY")
# process_docs_BAH = process_docs("BAH")
# process_docs_JOR = process_docs("JOR")
# process_docs_LIB = process_docs("LIB")
# process_docs_MAU = process_docs("MAU")
# process_docs_QAT = process_docs("QAT")
# process_docs_IRQ = process_docs("IRQ")
# process_docs_PAL = process_docs("PAL")
# process_docs_LEB = process_docs("LEB")
# process_docs_TUN = process_docs("TUN")
# process_docs_MSA = process_docs("MSA")
# process_docs_UAE = process_docs("UAE")
# process_docs_SUD = process_docs("SUD")
# process_docs_SYR = process_docs("SYR")
# process_docs_KUW = process_docs("KUW")
# process_docs_YEM = process_docs("YEM")



dialects = [
    'ALG', 'SAU', 'MOR', 'OMA', 'EGY', 'BAH', 'JOR', 'LIB', 'MAU', 'QAT',
    'IRQ', 'PAL', 'LEB', 'TUN', 'MSA', 'UAE', 'SUD', 'SYR', 'KUW', 'YEM'
]

# for dia in dialects:
#     globals()[f'process_docs_{dia}'] = process_docs(dia)



# load the BERTScore metric once globally
# bert_scorer = evaluate.load("bertscore")
# model_name = "intfloat/multilingual-e5-large"
# def bert_score(item):
#     """Compute BERTScore F1 for a single (reference, prediction) pair."""
#     reference, prediction = item
#     score = bert_scorer.compute(
#         predictions=[prediction],
#         references=[reference],
#         model_type=model_name,
#         num_layers=12
#     )
#     return score["f1"][0]

def bert_score(items): return items

def agg_bert_score(items):
    print("Computing BERT score...")
    # chose this model due to its small size compared to its good performance
    # on ArabicMTEB benchmark
    # src: https://arxiv.org/pdf/2411.01192
    model_name = "intfloat/multilingual-e5-large"
    bert_score = evaluate.load("bertscore")
    references, predictions = zip(*items)
    score = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=model_name,
        num_layers=24
    )
    return sum(score['f1']) / len(score['f1'])