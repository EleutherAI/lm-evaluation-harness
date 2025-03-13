from sklearn.preprocessing import MultiLabelBinarizer
import datasets

debug=False
TARGET_COLUMN = "Scheme" # apply dropna, Scheme is a valid attribute only for legal premises

MC_OPTIONS = ['Class', 'Itpr', 'Prec', 'Princ', 'Rule']

mlb = MultiLabelBinarizer()
mlb.fit([MC_OPTIONS])

# Funzione di debug
def debug_preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
    if debug:
        dataset = dataset.select([i for i in range(50)])      #Seleziono le prime n righe dal dataset invece di doverlo processare tutto    
    return dataset

def preprocess_scheme(dataset):
    # Invece di usare .dropna() propongo questo agli autori per evitare la conversione in DataFrame  dal momento che lm-eval lavora con i dataset huggingface
    if debug:
        print(f"Lunghezza originale: {len(dataset)}")
    dataset=dataset.filter(lambda x: x['Scheme'] is not None)
    if debug:
        print(f"Lunghezza filtrata: {len(dataset)}")
    dataset=debug_preprocess(dataset)
    if debug:
        print(f"Lunghezza per debug: {len(dataset)}")
    return dataset
