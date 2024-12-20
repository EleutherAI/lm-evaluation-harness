from sklearn.metrics import f1_score

def doc_to_text(doc) -> str:
  Context_concatenated= doc['Context_concatenated']
  Answer_concatenated = doc['Answer_concatenated']
  doc_to_text = f"Ti chiedo di risolvere un quesito. La lingua di questo quesito e' l'italiano."
  doc_to_text += f"\nTi daro' una lista di frasi (numerate da 1 a 7) che chiameremo **Contesto**, e un insieme di frasi (identificate da una lettera) che chiameremo **Risposte**.."
  doc_to_text += f"\nIl tuo compito e' di scegliere fra le **Risposte** la frase che potrebbe essere la frase seguente del **Contesto**."
  doc_to_text += f"\n# FORMATO: Devi mettere **SOLO** la lettera che corrisponde alla risposta migliore. Non inserire altro testo, ne' prima ne' dopo."
  doc_to_text += f"\n# DOMANDA"
  doc_to_text += f"\n**Contesto**"
  doc_to_text += f"\n{Context_concatenated}"
  doc_to_text += f"\n**Risposte**"
  doc_to_text += f"\n{Answer_concatenated}"
  doc_to_text += f"\n**La tua scelta**\n"
  return doc_to_text

def preprocess_dataset(dataset):
    dataset = dataset.select([i for i in range(1)])      # selecting 4 rows for DEBUG
    return dataset

def macro_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore