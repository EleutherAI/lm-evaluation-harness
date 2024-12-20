
def process_docs_abs(ds):
    ##Debug prendo solo 25 valori
    #ds = ds.select([i for i in range(25)])      # selecting 4 rows for DEBUG
    def _helper(doc):
        prompt = """Assegna un valore di astrazione da 1 a 5 alla parola {parola} nel contesto della frase seguente:
{frase}
Descrizione dei valori:
1 - La parola è estremamente concreta (e.g. un cane specifico)
2 - La parola è lievemente concreta (e.g. un cane di una certa razza)
3 - La parola è neutra (e.g. un cane tra tanti)
4 - La parola è lievemente astratta (e.g. un cane è un animale da compagnia)
5 - La parola è estremamente astratta (e.g. il cane è un mammifero)

Rispondi esclusivamente con il numero scelto senza aggiungere nessuna spiegazione
"""
        prompt = prompt.format(parola=doc["target_token"], frase=doc["text"])
        doc["prompt"] = prompt
        doc["abs"] = round(5 * doc["abs_mean"])
        if doc["abs"] == 5:
            doc["abs"] = 4
        doc["label"] = doc["abs"]
        doc["choices"] = ["1", "2", "3", "4", "5"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object


def process_docs_inc(ds):
        ##Debug prendo solo 25 valori
    #ds = ds.select([i for i in range(25)])      # selecting 4 rows for DEBUG
    def _helper(doc):
        prompt = """Assegna un valore di inclusività da 1 a 5 alla parola {parola} nel contesto della frase seguente:
{frase}
Descrizione dei valori:
1 - La parola è estremamente specifica (e.g. un cane specifico)
2 - La parola è lievemente specifica (e.g. un cane di una certa razza)
3 - La parola è neutra (e.g. un cane tra tanti)
4 - La parola è lievemente inclusiva (e.g. un cane è un animale da compagnia)
5 - La parola è estremamente inclusiva (e.g. il cane è un mammifero)

Rispondi esclusivamente con il numero scelto senza aggiungere nessuna spiegazione
"""
        prompt = prompt.format(parola=doc["target_token"], frase=doc["text"])
        doc["prompt"] = prompt
        doc["inc"] = round(5 * doc["inc_mean"])
        if doc["inc"] == 5:
            doc["inc"] = 4
        doc["label"] = doc["inc"]
        doc["choices"] = ["1", "2", "3", "4", "5"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

'''
def pearson(input):
    #a passthrough function for aggregate metrics
    return input
def pearson_agg(input):
    debug_=True
    from scipy.stats import pearsonr
    golds = ["".join(sublist[0]) for sublist in input]
    preds = [sublist[1][0] for sublist in input]
    if debug_:
        print("Etichette: \n")
        print(golds)
        print("Previsioni: \n")
        print(preds)
    #unzipped_list = list(zip(*items))
    #golds = doc["label"] #PER GPUCCE: IMPORTANTE CONTROLLARE SE È LA LABEL DESIDERATA
    #preds = generated[0]
    return pearsonr(golds, preds)[0]
    
    #return {"pearson": result}
# def list_fewshot_samples():
#     return {}
'''
