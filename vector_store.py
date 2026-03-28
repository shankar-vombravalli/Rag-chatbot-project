# Databricks notebook source
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []

    def load_data(self, folder):
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                self.texts.append(f.read())

    def create_index(self):
        embeddings = self.model.encode(self.texts)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def search(self, query, k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), k)

        results = [self.texts[i] for i in indices[0]]
        return results

# COMMAND ----------

