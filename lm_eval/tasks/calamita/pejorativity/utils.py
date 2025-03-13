import string
import unicodedata
import re
import random
import csv
#import os
def load_results(dataset):
    '''Funzione che recupera i dati salvati dall'iterazione precedente e li inserisce nella colonna 'pejorative' '''
    with open("pej_results.csv","r") as pej:
        golds=[]
        preds=[]  
        reader = csv.reader(pej)
        for row in reader:
            golds.append(int(row[0]))
            preds.append(int(row[1]))
    if dataset['pejorative']!=golds:
        print("FATAL: Golds from csv don't correspond to Gold labels in dataset!")
        exit()
    dataset=dataset.rename_column("pejorative","pejorative_gold")
    dataset=dataset.add_column('pejorative',preds)
    #os.replace("pej_results.csv","pej_results.csv.old")
    return dataset
def peggiorativa(doc):
    if int(doc['pejorative'])==0:
      return 'no'  
    else:
      return 'si'
    
def misogena(doc):
    if int(doc['misogyny'])==0:
      return 'no'  
    else:
      return 'si' 
  
def genera_misoginia(doc):
    if int(doc['pejorative'])==1:
        tipo='peggiorativa'
    elif int(doc['pejorative'])==0:
        tipo='non peggiorativa'
    else:
        tipo=''
        with open('error_log.log','a') as er:
           er.write(doc)
    return f"Istruzione: Considerando che la parola '{doc['word']}' è {tipo} in questa frase: '{doc['text']}', la frase è misogina?  Rispondi solo con 1 per sì, 0 per no. Risposta: "

def macro_f1_score_(items,task_type):
    from sklearn.metrics import f1_score
    unzipped_list = list(zip(*items))
    '''
    Il task "Pejorativity" prevede il salvataggio dei risultati ottenuti dall'esecuzione di questo task (parole preggiorative nel contesto) affinchè questi possano essere usati nel task successivo (misoginia). Il codice seguente salva i risultati nel file temporaneo "pej_results.txt"
    '''
    if task_type:
        with open("pej_results.csv","w") as pej:
            #pej.write("gold,pred\n")
            for gold,pred in items:
                pej.write(f"{gold},{pred}\n")
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore

def macro_f1_score_p(items):
    return macro_f1_score_(items,1)

def macro_f1_score_m(items):
    return macro_f1_score_(items,0)

def unisci_e_normalizza(stringa):
    #Unisco i caratteri e rimuovo gli spazi
    stringa=(''.join(stringa).strip(''))
    #Tolgo caratteri non ascii e accenti
    stringa=unicodedata.normalize('NFKD',stringa)
    stringa=''.join(filter(lambda x: x in string.ascii_letters,stringa))
    stringa.lower()
    return stringa
def exact_match_custom(dati):
     obiettivo=dati[0]
     risultato=dati[1]
     obiettivo=unisci_e_normalizza(obiettivo)
     risultato=unisci_e_normalizza(risultato)
     if obiettivo==risultato:
          return 1
     else:
          return 0

