# 🤖 RAG-Based Customer Support Assistant  
Using LangGraph & Human-in-the-Loop (HITL)

## 📌 Overview
This project is a **Retrieval-Augmented Generation (RAG) based Customer Support Assistant** designed to provide accurate, context-aware responses using a PDF knowledge base.

Unlike traditional chatbots, this system retrieves relevant information from documents and generates answers using a Large Language Model (LLM).

---

## 🚀 Features
- 📄 PDF-based knowledge ingestion  
- 🔍 Context-aware retrieval using embeddings  
- 🧠 LLM-powered response generation  
- 🔗 Workflow control using LangGraph  
- 🔀 Conditional routing based on query  
- 👨‍💻 Human-in-the-Loop (HITL) escalation  
- 💬 Interactive UI using Streamlit  

---

## 🏗️ System Architecture
The system consists of the following components:

- User Interface (Streamlit)
- Document Loader (PDF)
- Text Chunking Module
- Embedding Model
- Vector Database (ChromaDB)
- Retriever
- LLM (Answer Generator)
- LangGraph Workflow Engine
- HITL Module

---

## 🔄 Workflow
1. Upload PDF document  
2. Convert PDF to text  
3. Split text into chunks  
4. Generate embeddings  
5. Store embeddings in ChromaDB  
6. User submits query  
7. Retrieve relevant chunks  
8. Generate response using LLM  
9. Return answer to user  

---

## 🔀 Conditional Routing
- ✅ If relevant context found → Generate answer  
- ⚠️ If low confidence → Escalate  
- ❌ If complex query → Trigger HITL  

---

## 👨‍💻 Human-in-the-Loop (HITL)
- Activated when system cannot confidently answer  
- Query is escalated to human  
- Human response is returned to user  

---

## 🛠️ Tech Stack
- Python  
- LangChain  
- LangGraph  
- ChromaDB  
- HuggingFace / Groq LLM  
- Streamlit  

---

## 📂 Project Structure
project/
│── app.py
│── rag_pipeline.py
│── graph_flow.py
│── config.py
│── requirements.txt
│── README.md


---

## ▶️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

## Install Dependencies
pip install -r requirements.txt

## Run the Application
streamlit run app.py
