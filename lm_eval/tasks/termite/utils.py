import pandas as pd
import sqlalchemy
import numpy as np
from sqlalchemy import create_engine, MetaData
from sqlalchemy import text
from sqlalchemy import exc
from datasets import load_dataset
import datasets
base_path='./'
def debug(dataset: datasets.Dataset) -> datasets.Dataset:
#    dataset = dataset.select([i for i in range(25)])      # selecting 4 rows for DEBUG
    return dataset
def generate_prompt(doc):
    db_name=doc['db_id']
    file=base_path+'databases/'+db_name+'/no_insert_'+db_name+'.sql'
    try:
        with open(file,'r') as dbfile:
            sql_schema=dbfile.read()
    except:
        print("Errore fatale: file non trovato. Controllare 'base_path' nel file utils.py. Esecuzione terminata.")
        exit()  
    prompt="Considera il seguente database: \n"+sql_schema+"\n Traduci in SQL la seguente query in linguaggio naturale: \n"+doc['original']+' Restituisci SOLO la query senza alcuna spiegazione aggiuntiva. Penalità per ogni token non strettamente necessario'
    return prompt
def connect_to_db(db_name):

        #url="mysql+mysqlconnector://user:Password@127.0.0.1:3306/name_of_db"
        url="mysql+mysqlconnector://calamita:password@127.0.0.1:3306/{}".format(db_name)

        # Connect to b
        engine = create_engine(url,pool_pre_ping=True)

        connection = engine.connect()

        metadata = MetaData()

        metadata.reflect(bind=engine)

        return engine


##############################################################

def compare_results(cursor_result_1,cursor_result_2):   #compare gold query and generated query records (EXECUTION ACCURACY)
        
        #print(cursor_result_1.all() == cursor_result_2.all())

        if( cursor_result_1.all() == cursor_result_2.all() ):   #CursorResult.all() returns all result's rows in a list

            return 1

        else:

            return 0

def process_results(doc,generated):
    if True:
    #try:
        #print(f"Len original: {len(doc)}")
        #print(f"Len generated: {len(generated)}")
        #print(doc)
        #print(generated)
        result=0
        

        db=doc['db_id']
        database=db
        query_gold=doc['query']
        query_gen=generated[0]
        try: 
            query_gen='SELECT'+query_gen.split('SELECT')[1].split(';')[0]
        except:
            pass
        url="mysql+mysqlconnector://calamita:password@127.0.0.1:3306/{}".format(db)
        engine = create_engine(url)
        connection = engine.connect()
        metadata = MetaData()
        metadata.reflect(bind=engine)
#        print(f"Query gen prima di sqlalchemy: {query_gen}")
        try:
            escaped_sql_gold = sqlalchemy.text(query_gold)
            escaped_sql_gen = sqlalchemy.text(query_gen)
        except:
            return {"execution_accuracy": result}
#        print("DEBUG")
#        print(f"Query originale: {escaped_sql_gold}")
#       print(f"Query generata: {escaped_sql_gen}")
        with engine.connect() as conn:
            try:
                result_gold = conn.execute(escaped_sql_gold)
                #gestire query sbagliate sintatticamente
                try:
                        result_gen = conn.execute(escaped_sql_gen)
                        #print(f"Eseguo query: {escaped_sql_gen}")
                        result = compare_results(result_gold,result_gen)
                        #if (result==1):
                         #    print("Corrisponde")
                        #else:
                         #    print("Non corrisponde")
                except :
                        print("-----ERROR-----")
                        print("error on query:",query_gen)
                        print("db:",db)
                        print("---------------")
                        result = 0
                        print("Errore")
            except:
                    result = 0
                    print("Errore esecuzione")
    #except:
    #    result=0
    #    print("Errore generale")
    return {"execution_accuracy": result}



