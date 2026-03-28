# Databricks notebook source
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class RAG:
    def __init__(self):
        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # FAISS index
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store documents
        self.documents = []

    # ===============================
    # ADD DOCUMENT
    # ===============================
    def add_document(self, text):
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.documents.extend(chunks)

    # ===============================
    # SEARCH
    # ===============================
    def search(self, query, k=3):
        if len(self.documents) == 0:
            return None

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return "\n".join(results)

    # ===============================
    # OLLAMA QUERY (FIXED 🔥)
    # ===============================
    def query_ollama(self, prompt):
        url = "http://localhost:11434/api/generate"

        payload = {
            "model": "tinyllama",   # ✅ Use stable model
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=60)

            # HTTP error handling
            if response.status_code != 200:
                return f"❌ HTTP Error {response.status_code}: {response.text}"

            data = response.json()

            # Debug response structure
            if "response" not in data:
                return f"❌ Unexpected response: {data}"

            return data["response"]

        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Run: ollama serve"

        except requests.exceptions.Timeout:
            return "❌ Model timeout (system slow or heavy model)"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    # ===============================
    # ASK (MAIN FUNCTION)
    # ===============================
    def ask(self, query):
        context = self.search(query)

        if not context:
            prompt = f"""
You are a helpful AI assistant.

Answer clearly and simply.

Question:
{query}
"""
        else:
            prompt = f"""
You are a helpful AI assistant.

Answer ONLY based on the context below.

Context:
{context}

Question:
{query}
"""

        return self.query_ollama(prompt)