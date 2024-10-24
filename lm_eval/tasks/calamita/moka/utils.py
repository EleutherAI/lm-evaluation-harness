from datasets import Dataset
import datasets

def process_doc(dataset: datasets.Dataset):
    '''
    This function allows the adaptation of a JSON dataset, changing some names and types of columns,
    so that it is correctly processed during evaluation
    '''

    df = dataset.to_pandas()

    df.columns = [col.replace('-', '') for col in df.columns]

    columns_to_convert = ['Correct', 'Sibling', 'Close', 'Random']
    df[columns_to_convert] = df[columns_to_convert].astype(str)

    dataset = Dataset.from_pandas(df)
    
    return dataset



def process_results(doc, predictions):
    '''
    This function takes as input the loglikelihood's associated with the multiple choice options
    and allows the computation of the accuracy score and a further custom weighted error metric
    '''

    gold_label = int(doc['Correct'])

    pred_label = 1 + max(enumerate(predictions), key=lambda x: x[1][0])[0]
   
    acc = compute_accuracy(gold_label, pred_label)
    weighted_error = compute_weighted_error(doc, pred_label)
    
    
    return {"weighted_error": weighted_error, "acc": acc}

def compute_accuracy(gold_label, pred_label):
    return int(gold_label == pred_label)

def compute_weighted_error(doc, pred_label):
    
    # Define weights for each type of prediction
    weights = {
        'Correct': 0,   # No error
        'Sibling': 1,   # Slight error
        'Close': 2,     # More significant error
        'Random': 3     # Severe error
    }
    for key, weight in weights.items():
        if pred_label == int(doc[key]):
            return weight
    return max(weights.values())  # Assume maximum error if none match
    





