# Databricks notebook source
import streamlit as st
from src.rag_engine import RAG
import json
import os
import PyPDF2

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Shankar_AI Chatbot", layout="wide")

rag = RAG()

HISTORY_FILE = "chat_history.json"

# ===============================
# LOAD HISTORY
# ===============================
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        all_chats = json.load(f)
else:
    all_chats = {}

# ===============================
# SESSION STATE
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = "Chat 1"

# ===============================
# SAVE CHAT
# ===============================
def save_chat():
    all_chats[st.session_state.chat_id] = st.session_state.messages
    with open(HISTORY_FILE, "w") as f:
        json.dump(all_chats, f)

# ===============================
# PDF TEXT EXTRACTION
# ===============================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("💬 Chats")

# New Chat
if st.sidebar.button("➕ New Chat"):
    new_id = f"Chat {len(all_chats) + 1}"
    st.session_state.chat_id = new_id
    st.session_state.messages = []

# Load chats
for chat in all_chats.keys():
    if st.sidebar.button(chat):
        st.session_state.chat_id = chat
        st.session_state.messages = all_chats[chat]

# Search chats
search = st.sidebar.text_input("🔍 Search chats")

if search:
    st.sidebar.write("Results:")
    for chat, msgs in all_chats.items():
        if any(search.lower() in m["content"].lower() for m in msgs):
            st.sidebar.write(f"✅ {chat}")

# Clear chat
if st.sidebar.button("🗑 Clear Current Chat"):
    st.session_state.messages = []

# ===============================
# PDF UPLOAD
# ===============================
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        rag.add_document(text)
    st.sidebar.success("✅ PDF added!")

# ===============================
# MAIN UI
# ===============================
st.markdown(
    "<h1 style='text-align: center;'>🤖 Shankar_AI Chatbot</h1>",
    unsafe_allow_html=True
)

# Info message
if len(rag.documents) == 0:
    st.info("💡 Using general AI knowledge (no PDF uploaded)")
else:
    st.success("📚 Using uploaded documents")

# ===============================
# DISPLAY CHAT
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===============================
# INPUT
# ===============================
query = st.chat_input("Type your message...")

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag.ask(query)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save chat
    save_chat()

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🚀 Built by Shankar | Multi-Domain RAG Chatbot")