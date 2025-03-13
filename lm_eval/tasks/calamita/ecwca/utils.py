import evaluate
from typing import List, Dict
import nltk
import string
import unicodedata
from datasets import load_dataset
import datasets
debug=None


def debug_preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.select([i for i in range(30)])      # selecting 4 rows for DEBUG
    return dataset
    
def passt(item1):
     #a passthrough function for aggregate metrics
     return item1
def edit_distance(input):
    #a passthrough function for aggregate metrics
    return input
def words_avg_f1(input):
    #a passthrough function for aggregate metrics
    return input
    
def edit_distance_agg(input):
    correct_answers = ["".join(sublist[0]) for sublist in input]
    generations = [sublist[1][0] for sublist in input]
    edit_distance = 0.0
    for predicted_answer, target_answer in zip(generations, correct_answers):
        edit_distance += nltk.edit_distance(predicted_answer.replace(" ", ""), target_answer.replace(" ", ""))

    avg_edit_distance = edit_distance/len(correct_answers) if len(correct_answers) > 0 else 0.0
    if debug:
        with open("debug.log",'a') as dbg:
            dbg.write(f"----- DEBUG COMPLETO DI EDIT_DISTANCE AGG -----")
            dbg.write(f"-- Ho ricevuto {len(generations)} generations e {len(correct_answers)} correct answers --")
            dbg.write(f"-- Il computo dell'edit distance non diviso è {edit_distance}")
            dbg.write(f"-- Sto ritornando {avg_edit_distance} come risultato")
    return avg_edit_distance




def words_avg_f1_agg(input):
    correct_answers = ["".join(sublist[0]) for sublist in input]
    generations = [sublist[1][0] for sublist in input]
    def get_words(txt: str) -> List[str]:
        return txt.split()

    tot_f1 = 0.0
    for predicted_answer, target_answer in zip(generations, correct_answers):
        predicted_answer_words = set(get_words(predicted_answer))
        target_answer_words = set(get_words(target_answer))

        words_in_common = predicted_answer_words & target_answer_words

        r = len(words_in_common)/len(target_answer_words) if len(target_answer_words) > 0 else 0.0
        p = len(words_in_common)/len(predicted_answer_words) if len(predicted_answer_words) > 0 else 0.0
        f1 = 2*(r * p)/(r + p) if r+p > 0 else 0.0

        tot_f1 += f1

    avg_f1 = tot_f1/len(correct_answers) if len(correct_answers) > 0 else 0
    if debug:
        with open("debug.log",'a') as dbg:
            dbg.write(f"\n----- DEBUG COMPLETO DI WORDS_AVG_F1_AGG -----")
            dbg.write(f"\n-- Ho ricevuto {len(generations)} generations e {len(correct_answers)} correct answers --")
            dbg.write(f"\n-- Il computo dell'f1 non diviso è {tot_f1}")
            dbg.write(f"\n-- Sto ritornando {avg_f1} come risultato")
    return avg_f1

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
        if debug:
            with open("debug.log",'a') as dbg:
                dbg.write(f"\n----- DEBUG PARZIALE DI EXACT_MATCH_CUSTOM -----")
                dbg.write(f"\n{obiettivo} SI corrisponde a {risultato}")
        
        return 1
     else:
        if debug:
            with open("debug.log",'a') as dbg:
                dbg.write(f"\n----- DEBUG PARZIALE DI EXACT_MATCH_CUSTOM -----")
                dbg.write(f"\n{obiettivo} NON corrisponde a {risultato}")      
        return 0

