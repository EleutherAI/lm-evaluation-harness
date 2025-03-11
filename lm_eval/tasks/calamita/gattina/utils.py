#This code runs the two customs classifiers, NS and HA plus the SBERT score classifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import numpy as np
import torch
import re
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#ATTENZIONE RIMUOVERE QUANDO IN PRODUZIONE
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
ns_model = AutoModelForSequenceClassification.from_pretrained("mrinaldi/gattina-ns-classifier-fpt").to(device)
ns_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-uncased")
ha_model = SentenceTransformer("mrinaldi/mrinaldi/gattina-ha-classifier-cossim-ffpt").to(device)
sbert_model = SentenceTransformer("nickprock/sentence-bert-base-italian-xxl-uncased").to(device)


def preprocess_text(text):
    result = text.strip()
    chars_to_remove = '"*#[](){}/<>™®©℠¢£¤¥€×÷±§¶†‡•¦¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿×'+"'"
    while result and (result[0] in chars_to_remove ):
        result = result[1:]
    while result and (result[-1] in chars_to_remove ):
        result = result[:-1]  
    titolo_pattern = r'^Titolo:\s*"(.+)"$'
    titolo_match = re.match(titolo_pattern, result.strip())
    if titolo_match:
        result = titolo_match.group(1)
    result = result.split('\n')[0]
    return result.strip()

def get_ns_score(headline):
    # Tokenize the title
    inputs = ns_tokenizer(
        headline,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    device = next(ns_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = ns_model(**inputs).logits
        score = torch.sigmoid(outputs).item()
    #if score<0: #Clip in case of negative score
    #    score=0
    #if score>1:
    #    score=1
    score = 1-score #In this way higher is better
    return score
def get_ha_score(article,headline):
    article_embed = ha_model.encode(article, convert_to_tensor=True)
    headline_embed = ha_model.encode(headline, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(article_embed, headline_embed)
    return cosine_score.cpu().item()
def get_sbert_score(original,generated):
    original_embed = sbert_model.encode(original, convert_to_tensor=True)
    generated_embed = sbert_model.encode(generated, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(original_embed, generated_embed)
    return cosine_score.cpu().item()   


def preprocess_docs(dataset):
	def clean_text(item):
		item['title']=preprocess_text(item['title'])
	dataset.map(clean_text)
	return dataset

def genera_prompt(doc):
	prompt="""
	Il tuo compito è generare un titolo accattivante e informativo per l'articolo fornito.
        Requisiti:
           - Titolo breve
           - Cattura l'essenza dell'articolo
           - Usa un linguaggio vivido e coinvolgente
           - Non generare alcun tipo di testo che non sia il titolo dell'articolo
           - Usa esclusivamente l'Italiano.
         Presta particolare attenzione ai seguenti titoli di esempio e adotta lo stesso stile:
         
	"""
	titoli_esempio=[
	"Nella Via Lattea c'è un oggetto misterioso, è velocissimo",
    "Nasce il gemello digitale del rischio ambientale in Italia",
    "I cinque modi in cui il cervello invecchia",
    "Covid-19, il mistero degli over 90",
    "A 44 e a 60 anni i due gradini chiave dell'invecchiamento",
    "Palestra o snack? la scelta dipende da un messaggero chimico",
    "Dagli stadi alle spiagge, sono i salti a sincronizzare il ballo",
    "Dalle rose alle melanzane, ecco i geni delle spine",
    "Così il Covid accelera l'invecchiamento",
    "Uno zucchero naturale contro la calvizie, bene i test sui topi",
    "Scoperto nel cervello il circuito dell'effetto placebo",
    "Pronto il Google Earth del cuore umano",
    "Una molecola può ringiovanire il sistema immunitario",
    "Scoperto il dizionario dei sinonimi e contrari del cervello",
    "Le farfalle nello stomaco non sono solo un modo di dire",
    "Pronto il primo orologio nucleare, il più preciso del mondo",
    "Gli uccelli in volo si comportano come gli atomi",
    "L'Italia ritenta la sfida impossibile della geometria",
    "Le auto nel traffico come i batteri in cerca di cibo",
    "Robot come alleati, trovata la chiave per collaborare con gli umani",
    "Dalle spugne di vetro grattacieli più sottili e resistenti",
    "L'IA non è razionale, fa ragionamenti non logici"]
	for titolo_esempio in random.choices(titoli_esempio,k=5):
		prompt+=titolo_esempio+'\n'
	prompt+=doc['text']
	return prompt

def process_results(doc,generated):
	originale=preprocess_text(doc['title'])
	articolo=doc['text']
	generato=preprocess_text(generated[0])
	return {
		"sbert_score": get_sbert_score(originale,generato),
		"ns_score": get_ns_score(generato),
		"ha_score": get_ha_score(articolo,generated[0])
	}

	#MANCA ROUGEL
