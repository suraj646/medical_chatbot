# 🩺 Medical RAG Chatbot (Streamlit + LangChain + FAISS)

A **Retrieval-Augmented Generation (RAG)** based Medical Chatbot built using **Streamlit**, **LangChain**, **FAISS**, and **HuggingFace LLMs**.

This chatbot answers medical questions using only a custom knowledge base and also shows the **source documents** used for each response.

---

## 🚀 Project Overview

The **Medical RAG Chatbot** is designed to provide accurate and transparent answers by combining:

- **Document Retrieval** (FAISS Vector Search)
- **LLM Response Generation**
- **Chat Memory** (Session-based history)
- **Source References** (Documents used for answers)

---

## ✨ Features

✅ Medical Question Answering using RAG  
✅ FAISS Vector Database for fast retrieval  
✅ HuggingFace LLM Integration   
✅ Session-based Chat History (Memory)  
✅ Source Display for Transparency  
✅ Streamlit Chat UI  

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – Web Interface
- **LangChain** – RAG Pipeline
- **FAISS** – Vector Similarity Search
- **HuggingFace Endpoints** – LLM + Embeddings
- **Sentence Transformers** – Embedding Model
- **dotenv** – Environment Variable Management

---

## 📂 Project Structure

```bash
medical_chatbot/
│
├── chatbot.py                  # Main Streamlit chatbot app
├── vectorstore/
│   └── db_faiss/               # Stored FAISS vector database
│
├── data/                       # Medical documents used as knowledge base
│
├── .env                        # HuggingFace API key (not uploaded to GitHub)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
