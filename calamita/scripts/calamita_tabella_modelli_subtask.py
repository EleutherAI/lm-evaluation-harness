#!/usr/bin/env python
# coding: utf-8

# In[46]:


import json
get_ipython().system('pip install pandas')
import pandas as pd
import requests
import os


# In[57]:


# Raw URL of the file on GitHub
# Premere il tasto 'raw' durante visualizzazione del file su github
raw_url = 'https://raw.githubusercontent.com/CALAMITA-AILC/CALAMITA-harness/refs/heads/main/calamita/calamita_tasks_v1.txt'

response = requests.get(raw_url)
if response.status_code == 200:
    file_content = response.text
    lista_elementi = file_content.split('\n')
    lista_subtasks = [elemento.strip() for elemento in lista_elementi if elemento.strip()]
else:
    raise Exception(f"Failed to retrieve the file. Status code: {response.status_code}\nCopia il file con l'elenco delle subtask in altro modo")


# In[54]:


# da modificare se necessario
PATH_CARTELLA_RESULTS = '.'

if 'results_calamita_2024' not in os.listdir(PATH_CARTELLA_RESULTS):
    raise Exception("path da modificare")


# In[55]:


# le directory contengono il nome del modello
modelli = [dir.split('__')[1] for dir in os.listdir('./results_calamita_2024') if dir not in ['README.md','.git','.gitattributes']]
modelli


# In[89]:


# creazione del dictionary da cui verrà creata la tabella finale
# istanziamo tutti no e poi passiamo a yes man mano
tabellaFinale = {el:{model:"no" for model in modelli} for el in lista_subtasks}
#tabellaFinale


# In[104]:


DICT_RENAMED = {'conflict_detect': 'gita_conflict_detect', 'physical_state':'gita_physical_state', 'story_class':'gita_story_class'}
LISTA_IGNORE = ['agr_tasks', 'od_tasks', 'caus_tasks'] #per la sottotask a due livelli di blm


# In[105]:


# per ogni directory di modello
modelsDirectories = [dir for dir in os.listdir('./results_calamita_2024') if dir not in ['README.md','.git','.gitattributes']]
for modelIndex in range(len(modelli)):
    modelDirectory = modelsDirectories[modelIndex]
    modelDirectoryPath = PATH_CARTELLA_RESULTS+'/results_calamita_2024/'+modelDirectory
    # selezionare le sotto directory contenenti results e non samples
    results = [dir for dir in os.listdir(modelDirectoryPath) if dir[0:7] == 'results']
    # iterare tutti i file results
    for resultFile in results:
        resultsFilePath = modelDirectoryPath+'/'+resultFile
        with open(resultsFilePath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        # estrarre elenco subtasks testate tramite il campo 'group_subtasks'
        # se il value corrispondente è lista vuota: è subtask
        # else, è task. in quel caso va estratta la lista value contenente le subtasks
        group_subtasks = data['group_subtasks'].keys()
        subtasksResult = []
        for subt in group_subtasks:
            if len(data['group_subtasks'][subt]) == 0:
                subtasksResult.append(subt)
            else:
                subtasksResult.extend(data['group_subtasks'][subt])
        # estratte le sottotask, si aggiorna la tabella
        for subt in subtasksResult:
            try:
                # possibile errore, sotto task non nella lista originale
                if subt not in lista_subtasks:
                    raise Exception(f"subtask '{subt}' non presente nell'elenco delle task, trovata nel file: {resultsFilePath}",subt,resultsFilePath)
                else:
                    modello = modelli[modelIndex]
                    tabellaFinale[subt][modello] = "yes"
            except Exception as e:
                error_message, subtaskError, fileError = e.args
                if subtaskError in LISTA_IGNORE:
                    pass
                elif subtaskError in DICT_RENAMED.keys():
                    subtask_newname = DICT_RENAMED[subtaskError]
                    modello = modelli[modelIndex]
                    tabellaFinale[subtask_newname][modello] = "yes"
                else:
                    raise Exception(e.args)


# In[90]:


#tabellaFinale


# In[106]:


# Convertire in DataFrame
df = pd.DataFrame.from_dict(tabellaFinale, orient="index")
df.index.name = "Subtask"

# Salvare in CSV
csv_filename = "tasks_done.csv"
df.to_csv(csv_filename)


# # Invece, se si parte da file parsed.json

# In[112]:


# Raw URL of the file on GitHub
# Premere il tasto 'raw' durante visualizzazione del file su github
raw_url = 'https://raw.githubusercontent.com/CALAMITA-AILC/CALAMITA-harness/refs/heads/main/calamita_utilities/parsed.json'

response = requests.get(raw_url)
if response.status_code == 200:
    file_content = response.text
    parsed = json.loads(file_content)

else:
    raise Exception(f"Failed to retrieve the file. Status code: {response.status_code}")


# In[113]:


# Print the lengths of each key's value (useful for debugging)
for key, value in parsed.items():
    print(f"Length of '{key}': {len(value)}")


# In[115]:


# metodo per ricondurre il file parsed.json in un csv ordinato
# struttura input:
# [
#  {'subtask_name': 'hellaswag_ita',
#   'belongTo': 'itaeval',
#   'evaluations': [
#     {'model_name': 'meta-llama/Llama-3.1-8B-Instruct',
#      'metrics': [
#       {'metric': 'acc', 'result': 0.4532961561441944},
#       {'metric': 'acc_norm', 'result': 0.5792670782712607}]},
#     {'model_name': ...
#     }
#   ]
#  },
#  {'subtask_name': ...
#  }
# ]
#
# struttura output:
# [{
#    "subtask" : ,
#    "models": ["",""],
#   },
#   ..
# ]

listaDati = []
for subtask in parsed['subtask']:
  subtaskName = subtask['subtask_name']
  models = []
  for evaluation in subtask['evaluations']:
      models.append(evaluation['model_name'])
  listaDati.append({'subtask': subtaskName, 'models': models})
#listaDati


# In[117]:


# estrarre i modelli usati dai dati del metodo precedente
models = sorted({model.split('/')[1] for task_details in listaDati for model in task_details['models']})
models


# In[118]:


# output structure
# {'subtask': {
#    'model': 'yes',
#    'model2': 'yes',
#    'model3': 'yes',
#    'model4': 'yes'
#   },
#  'subtask2' ..
#  ..
#  }

table = {
    entry['subtask']: {model: "yes" if model in entry['models'] else "no" for model in models}
    for entry in listaDati
}


# In[ ]:


# Convertire in DataFrame
df = pd.DataFrame.from_dict(table, orient="index")
df.index.name = "Subtask"

# Salvare in CSV
csv_filename = "tasks_done.csv"
df.to_csv(csv_filename)

