# Databricks notebook source
import os
from PyPDF2 import PdfReader
import nltk

nltk.download('punkt')

def extract_text(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def chunk_text(text, chunk_size=200, overlap=50):
    from nltk.tokenize import word_tokenize

    words = word_tokenize(text)
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def process_documents(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            path = os.path.join(input_dir, file)

            text = extract_text(path)
            text = clean_text(text)

            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                with open(os.path.join(output_dir, f"{file}_{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(chunk)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------