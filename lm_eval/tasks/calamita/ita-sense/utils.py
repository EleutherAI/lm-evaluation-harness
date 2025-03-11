from evaluate import load
from rouge_score import rouge_scorer
from datasets import load_dataset
import datasets
import string
import unicodedata
import re
from evaluate import load
from rouge_score import rouge_scorer
import traceback 
ROUGE_SCORER = None
BERT_SCORER = None
#      METRIC_LIST = ["harmonicRougeBertScore", "rougeL", "bertScore"]
def debug(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.select([i for i in range(25)])      # selecting 4 rows for DEBUG
    return dataset


def process_results_gen(doc, results):
    try:
        expected_output = doc["output"].strip()
        result = results[0][0].strip()

        global ROUGE_SCORER
        if ROUGE_SCORER is None:
            ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        
        global BERT_SCORER
        if BERT_SCORER is None:
            BERT_SCORER = load("bertscore", keep_in_memory=True)

        rouge_result = ROUGE_SCORER.score(expected_output, result)['rougeL'].fmeasure
        bert_result = BERT_SCORER.compute(predictions=[result], references=[expected_output], lang="it")["f1"][0]
        if (rouge_result+bert_result!=0):
            rouge_bert_result = (5 * rouge_result * bert_result) / (4 * rouge_result + bert_result)
        else:
            rouge_bert_result=0
    except:
            traceback.print_exc() 
            rouge_bert_result=0
            rouge_result=0
            bert_result=0
    return {"rougeBertScore": rouge_bert_result, "rougeL": rouge_result, "bertScore": bert_result}



def extract_answer(doc,results):
    try:
        target=int(doc['output'])
        result=results[0][0].strip(' ').strip('\n')
        try:
            result=re.findall('[0-9]+', result)[0]
            result=int(result)
            #print(f"DEBUG-Filtrata: {result}")
        except:
            #print(f"Nessun numero in {result}, {results}")
            return {"extract_answer": 0}
        if (target==result):
            return {"extract_answer": 1}
        else:
            return {"extract_answer": 0}
    except:
        return {"extract_answer": 0}

