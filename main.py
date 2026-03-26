# Databricks notebook source
from src.rag_engine import RAG
import time

rag = RAG()

print("="*50)
print("🤖 RAG Chatbot (type 'exit' to quit)")
print("="*50)

while True:
    query = input("\n🧑 You: ")

    if query.lower() == "exit":
        print("👋 Exiting chatbot...")
        break

    print("\n🤖 Bot is thinking...", end="")
    time.sleep(1)

    answer = rag.ask(query)

    print("\r🤖 Bot:", answer)