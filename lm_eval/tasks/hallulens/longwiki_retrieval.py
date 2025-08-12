# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import sqlite3
import numpy as np
import pickle as pkl
from typing import List

from lm_eval.tasks.hallulens.cache import Cache
from lm_eval.tasks.hallulens.retrieval import DocDB
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class LongWikiDB(DocDB):
    def __init__(self, db_path:str, data_path:str = None):
        # import DocDB from FactScore
        super(LongWikiDB, self).__init__(db_path, data_path)
        self.title_db_path = db_path.replace(".db", "-title.db")
        self.title_connection = sqlite3.connect(self.title_db_path, check_same_thread=False)
        self.SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

    def get_relevant_titles(self, entity: str):
        cursor = self.title_connection.cursor()
        entity = entity.replace("'", "''")
        cursor.execute("SELECT title_name FROM titles WHERE title_name LIKE ?", ('%' + entity + '%',))
        results = cursor.fetchall()
        cursor.close()
        results = [r[0] for r in results]
        return results
    
    def get_whole_passages(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT title, text FROM documents")
        results = cursor.fetchall()
        results = [r for r in results]
        results = [{"title": r[0], "text": para} for r in results for para in r[1].split(self.SPECIAL_SEPARATOR)]
        return results
    

class LongWikiRetrieval(object):
    def __init__(self, db, cache_base_path, embed_cache_path,
                 retrieval_type="gtr-t5-large", batch_size=None, debugging=False):
        
        self.db = db
        self.CACHE_BASE_PATH = cache_base_path
        self.embed_cache_path = embed_cache_path
        self.load_cache()

        self.retrieval_type = retrieval_type
        self.batch_size = batch_size

        ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
        self.ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple", batch_size=8, device=0)
        
        self.encoder = None
        self.not_existing_pages = set()
        self.debugging = debugging
    
    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type, device=0)
        self.encoder = encoder
        assert self.batch_size is not None
    
    def load_cache(self):
        # cache
        self.relevant_pages_cache_path = f"{self.CACHE_BASE_PATH}/relevant_pages_cache.json" # key: entity, value: list of relevant page titles
        self.Q_NER_cache_path = f"{self.CACHE_BASE_PATH}/question_to_ner_cache.json"
        self.cache_path =f"{self.CACHE_BASE_PATH}/cache.json"
        self.embed_cache_path = self.embed_cache_path

        self.add_n = 0
        self.add_n_embed = 0

        # embedding cache
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}

        # question to NER cache
        self.Q_NER_cache = Cache(self.Q_NER_cache_path)

        # single entity to relevant pages cache
        self.relevant_pages_cache = Cache(self.relevant_pages_cache_path)

        # prompt query to top-5 passage cache 
        self.cache = Cache(self.cache_path)

    
    def save_cache(self):        
        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)
                
            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)
        
    def get_topk_passages(self, topic, retrieval_query, key_passages, k=5):
        if self.encoder is None:
            self.load_encoder()

        self.add_n_embed = 0
        passage_vectors_all = None
        passages_all = []

        for topic, passages in key_passages.items():
            passages_all += passages

            if topic in self.embed_cache:
                passage_vectors = self.embed_cache[topic]
            else:
                inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
                passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device, clean_up_tokenization_spaces=True)
                self.embed_cache[topic] = passage_vectors
                
                self.add_n_embed += 1

            passage_vectors_all = np.concatenate([passage_vectors_all, passage_vectors], axis=0) if passage_vectors_all is not None else passage_vectors

        query_vectors = self.encoder.encode([retrieval_query], 
                                            batch_size=self.batch_size,
                                            device=self.encoder.device)[0]
        
        scores = np.inner(query_vectors, passage_vectors_all)
        indices = np.argsort(-scores)[:k]

        if self.add_n_embed > 0:
            self.save_cache()

        return [passages_all[i] for i in indices]

    def make_ner_cache(self, questions: List[str]):
        for question in questions:
            ners = self.Q_NER_cache.get_item(question)
            if ners is None:
                ner_results = self.ner(question)
                ners = [r['word'] for r in ner_results if '#' not in r['word']]
                self.Q_NER_cache.set_item(question, ners)
                
    def get_topk_related_passages(self, topic, claim, question, k=5, use_cache=True):
        """
            NER based top-k passage retrieval
            Extract named entities from question, get relevant pages for each entity.
            -> Passage pool: topic (where question is generated), NERs, NER-relevant pages
            Out of the passage pool, get top-k similar passages to the query using the encoder (self.get_topk_passages)
            return top-k passages
        """
        #### Function called from facthalu.py
        retrieval_query = topic + " " + claim.strip()
        cache_key = topic + "#" + claim.strip()

        # check cache
        cache_res = self.cache.get_item(cache_key)
        if use_cache and cache_res is not None:
            return cache_res
        
        # Using NER to get named entities from question
        ners, ner_relevant_titles = [], []
        ners = self.Q_NER_cache.get_item(question)
        for ner in ners:
            pgs_selected = self.relevant_pages_cache.get_item(ner)
            if pgs_selected:
                pgs = pgs_selected
            else:
                pgs = self.db.get_relevant_titles(ner)
                if not pgs: 
                    continue
                self.relevant_pages_cache.set_item(ner, pgs)
            ner_relevant_titles += [pg for pg in pgs if ((pg.lower() in claim.lower()) or (pg.lower() in question.lower()))]
        
        # Get all relevant pages
        combined = [topic] + ner_relevant_titles + ners
        all_related_pages = list(set(combined))

        # get all passages
        key_passages = {}
        for title in all_related_pages:
            title = title.replace("_", " ")
            if title in self.not_existing_pages: continue
            try:
                pages = self.db.get_text_from_title(title)
                key_passages[title] = pages
            except:
                self.not_existing_pages.add(title)
                continue
        
        top_k_related_passages = self.get_topk_passages(topic, retrieval_query, key_passages, k)
        self.cache.set_item(cache_key, top_k_related_passages)

        self.add_n += 1
        return top_k_related_passages