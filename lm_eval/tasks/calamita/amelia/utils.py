import json 
from sklearn.metrics import f1_score
import datasets
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer



debug=False

# Funzioni di preprocessing per eliminare, in Scheme e Premise Type, i documenti in cui manca la colonna di riferimento
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

def preprocess_premisetype(dataset):
    # Invece di usare .dropna() propongo questo agli autori per evitare la conversione in DataFrame  dal momento che lm-eval lavora con i dataset huggingface
    if debug:
        print(f"Lunghezza originale: {len(dataset)}")
    dataset=dataset.filter(lambda x: x['Type'] is not None)
    if debug:
        print(f"Lunghezza filtrata: {len(dataset)}")
        dataset=debug_preprocess(dataset)
    if debug:
        print(f"Lunghezza per debug: {len(dataset)}")
    return dataset

# Funzione di debug
def debug_preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
    if debug:
        dataset = dataset.select([i for i in range(30)])      #Seleziono le prime n righe dal dataset invece di doverlo processare tutto    
    return dataset


# Funzioni passthrough per le metriche f1 da fare con aggregazione
def macro_f1_score(input):
    return input

def f1_classes(input):
    return input
# Fine funzioni passthrough


# Funzione proposta da discutere con autori
def extract_first_prem_conc(text):
        cleaned_text = text.strip('\n . ( )')
        # Estrae la prima parola
        first_word_match = re.match(r'\w+', cleaned_text, re.IGNORECASE)      
        if first_word_match:
            first_word = first_word_match.group().lower()
            
            if first_word in ['prem', 'premessa']:
                return 'prem'
            elif first_word in ['conc', 'conclusione']:
                return 'conc'    
        return 'no_match'

def extract_list(text,mc_options):
    print(text)
    cleaned_text = text.strip('\n . ( ) \\')
    match = re.search(r'\[(.*?)\]', cleaned_text)
    if match:
        list_content = match.group(1)
        try:
            extracted_list = eval(f'[{list_content}]')
            #Prima lettera maiuscola
            uppercase_list = [element.capitalize() for element in extracted_list]
            for i in uppercase_list:
                if mc_options is not None:
                    if i not in mc_options:
                        if debug:
                            print(f"Scarto risultato. {i} non è presente in {mc_options}")
                        return [None]
            return uppercase_list
        except:
            if debug:
                print("Lista inconvertibile.")
            return [None]
    if debug:
        print("Non ho trovato liste")
    return [None]

    
# Funzioni originali degli autori
def macro_f1_score_func(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def f1_classes_func(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None) # returns array with the scores for each class


# Funzioni di aggregazione che vengono richiamate direttamente dallo YAML [PER IL PROBLEMA BINARIO]
def macro_f1_score_agg(dati):
    correct_answers = np.array(["".join(sublist[0]) for sublist in dati])
    generations = np.array([extract_first_prem_conc(sublist[1][0]) for sublist in dati])
    #print (correct_answers)
    #print (generations)
    return macro_f1_score_func(correct_answers,generations)

def f1_classes_agg(dati):
    correct_answers = np.array(["".join(sublist[0]) for sublist in dati])
    generations = np.array([extract_first_prem_conc(sublist[1][0]) for sublist in dati])
    #print (correct_answers)
    #print (generations)
    return str(f1_classes_func(correct_answers,generations))



# Funzioni di aggregazione che vengono richiamate direttamente dallo YAML [PER IL PROBLEMA MULTICLASSE]
def macro_f1_score_agg_MC(dati,mlb,mc_options):
    correct_answers = np.array([mlb.transform([json.loads(bytes(sublist[0]))])[0] for sublist in dati])
    generations = np.array([mlb.transform([extract_list(sublist[1][0],mc_options)])[0] for sublist in dati])
        
    #print(correct_answers)
    #print(generations)

    return macro_f1_score_func(correct_answers,generations)

def f1_classes_agg_MC(dati,mlb,mc_options):
    correct_answers = np.array([mlb.transform([json.loads(bytes(sublist[0]))])[0] for sublist in dati])
    generations = np.array([mlb.transform([extract_list(sublist[1][0],mc_options)])[0] for sublist in dati])
        
    #print(correct_answers)
    #print(generations)

    return str(f1_classes_func(correct_answers,generations))


def macro_f1_score_agg_MC_scheme(dati):
    TARGET_COLUMN = "Scheme" # apply dropna, Scheme is a valid attribute only for legal premises
    MC_OPTIONS = ['Class', 'Itpr', 'Prec', 'Princ', 'Rule']
    mlb = MultiLabelBinarizer()
    mlb.fit([MC_OPTIONS])
    return macro_f1_score_agg_MC(dati,mlb,MC_OPTIONS)

def f1_classes_agg_MC_scheme(dati):
    TARGET_COLUMN = "Scheme" # apply dropna, Scheme is a valid attribute only for legal premises
    MC_OPTIONS = ['Class', 'Itpr', 'Prec', 'Princ', 'Rule']
    mlb = MultiLabelBinarizer()
    mlb.fit([MC_OPTIONS])
    return f1_classes_agg_MC(dati,mlb,MC_OPTIONS)



def macro_f1_score_agg_MC_premisetype(dati):
    TARGET_COLUMN = "Type" # apply dropna, Scheme is a valid attribute only for legal premises
    MC_OPTIONS = ["F", "L"]
    mlb = MultiLabelBinarizer()
    mlb.fit([MC_OPTIONS])
    return macro_f1_score_agg_MC(dati,mlb,MC_OPTIONS)

def f1_classes_agg_MC_premisetype(dati):
    TARGET_COLUMN = "Type" # apply dropna, Scheme is a valid attribute only for legal premises
    MC_OPTIONS = ["F", "L"]
    mlb = MultiLabelBinarizer()
    mlb.fit([MC_OPTIONS])
    return f1_classes_agg_MC(dati,mlb,MC_OPTIONS)

# Funzione aggiunta dal revisore. Può essere utile, costo aggiuntivo irrilevante. Discutere con autori.
def accuracy(doc):
    gold="".join(doc[0])
    target=extract_first_prem_conc(doc[1][0])
    return gold==target

def MC_accuracy(doc):
    lista_gold=json.loads(bytes(doc[0]))
    lista_gen=extract_list(doc[1][0],None)
    #print(f"\nLista gold: {lista_gold} Lista gen: {lista_gen}")
    if lista_gen==None:
        return 0
    try:
        if len(lista_gold) != len(lista_gen):
            return 0
    except:
        return 0
    return 1 if sorted(lista_gold) == sorted(lista_gen) else 0

