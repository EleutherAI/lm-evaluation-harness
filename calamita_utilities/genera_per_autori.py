import csv
import os
import shutil
# Lo script è sporco: i nomi di alcuni file e cartelle sono delle costanti
csv_tabellone='./tabellaConversioneGruppiTask.csv'
cartella_risultati='tests_calamita_latest'
decennio="_202"
outputpath="per_autori"
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
# Carico il dizionario di conversione Sottotask - Task
with open(csv_tabellone,'r') as f:
     conversione=dict()
     reader=csv.reader(f)
     for row in reader:
        conversione[row[0]]=row[1]
# Tiro fuori il nome del sottotask dal nome del file
def recupera_nome(nomefile):
    sottotask=nomefile.split('samples_')[1].split(decennio)[0]
    return sottotask
# Carico i nomi dei modelli e la lista di file presenti
listamodelli=set()
listafile=set()
for item in os.walk(cartella_risultati):
    for modelli in item[1]:
        listamodelli.add(modelli)
    # Recupero la lista dei sottotask in modo da verificare in seguito che il tabellone li contenga tutti
    for file in item[2]:
        if file.startswith('samples_'):
            # Questa cosa è sporca.
            # Presuppone che la valutazione sia stata fatta negli anni Venti del XXI secolo (il che è ok)
            # ma attenzione, non possono esistere sottotask che contengono _202 nel nome!   
            listafile.add(recupera_nome(file))   
# Primo controllo: tutti i sottotask che hanno samples sono nel tabellone?
mancanti=False
with open("report_presenze_sottotask.txt","a") as reportfile:
    reportfile.write("----------\n")
    for sottotask in listafile:
        if sottotask not in conversione.keys():
            reportfile.write(f"{sottotask} non è nel tabellone\n")  
            mancanti=True
    reportfile.write("----------\n") 
if mancanti:
    if not os.path.exists(os.path.join(outputpath,'orfani')):
        os.makedirs(os.path.join(outputpath,'orfani'))                    
# Secondo controllo: ciascun sottotask è stato provato su tutti i modelli?
with open("report_modelli_sottotask.txt", "a") as reportfile:
    reportfile.write("----------\n")
    for sottotask in listafile:
        for modello in listamodelli:
            path_atteso = os.path.join(cartella_risultati, modello)
            trovato = False
            for item in os.walk(path_atteso):
                for file in item[2]:
                    if file.startswith('samples_') and recupera_nome(file) == sottotask:
                        trovato = True
                        break
                if trovato:
                    break
            if not trovato:
                reportfile.write(f"Il sottotask {sottotask} non è stato testato sul modello {modello}\n")
    reportfile.write("----------\n")
#Creo una cartella per ciascun task
tasks=set(conversione.values())
for task in tasks:
    path=os.path.join(outputpath,task)
    if not os.path.exists(path):
        os.mkdir(path)
#... E sottotask
for sottotask in conversione.keys():
   path=os.path.join(outputpath,conversione[sottotask],sottotask)
   if not os.path.exists(path):
        os.mkdir(path)
#Finalmente copiamo i file 
for modello in listamodelli:
    for item in os.walk(os.path.join(cartella_risultati,modello)):
        for file in item[2]:
            if file.startswith('samples_'):
                data=decennio+file.split(decennio)[1]
                sottotask=recupera_nome(file)
                nuovonome=f"{modello}_{data}.jsonl"
                path_sorgente=os.path.join(cartella_risultati,modello,file)
                if sottotask in conversione.keys():
                    path_destinazione=os.path.join(outputpath,conversione[sottotask],sottotask,nuovonome)
                else:
                    
                    path_destinazione=os.path.join(outputpath,'orfani',sottotask+'_'+nuovonome)
                shutil.copyfile(path_sorgente,path_destinazione)
