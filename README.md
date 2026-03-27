# 🤖 RAG Chatbot using LLM

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that answers user queries based on custom documents.

Instead of relying only on LLM knowledge, the system retrieves relevant information from a document store and generates accurate responses.

---

## ⚙️ Tech Stack

* Python
* Sentence Transformers (Embeddings)
* FAISS / ChromaDB (Vector Store)
* Ollama / LLM
* Streamlit (User Interface)

---

## 🧠 How It Works (Pipeline)

1. Load documents (PDF/Text)
2. Preprocess text
3. Convert text → embeddings
4. Store embeddings in vector database
5. User asks question
6. Retrieve relevant documents
7. Send context + question to LLM
8. Generate final answer

---

## 📁 Project Structure

```bash
Rag-chatbot-project/
│
├── app.py                # Streamlit UI
├── rag_engine.py         # RAG pipeline logic
├── vector_store.py       # Embedding + storage
├── preprocessing.py      # Data cleaning
├── requirements.txt      # Dependencies
│
├── data/                 # Input documents
├── images/               # Screenshots
```

---

## 🚀 How to Run

### Step 1: Install dependencies

pip install -r requirements.txt

### Step 2: Run application

python app.py

---


---

## 🎯 Key Features

* Semantic search using embeddings
* Context-aware responses
* Efficient document retrieval
* Simple UI using Streamlit

---

## 🔥 Future Improvements

* Add multi-document support
* Improve UI design
* Deploy using cloud (AWS/Render)
* Add conversation memory

---

## 👨‍💻 Author

**Shankar**
Aspiring AI/ML Engineer

<img width="1100" height="822" alt="output_2 (2)" src="https://github.com/user-attachments/assets/767d20ed-6a57-4517-8e8a-3823836a5ad8" />

