# 📚 RAG PDF Mini Project

A lightweight, local Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents. This project leverages **Ollama** for local LLM inference and **FAISS** for efficient similarity search.



## 🚀 Overview
This application transforms static PDF documents into a searchable knowledge base. By converting text into high-dimensional vectors, it allows an LLM to "read" your specific file and provide answers grounded strictly in the provided context.

### Key Features
* **Local Inference:** Uses `ChatOllama` with the **Mistral** model for privacy and cost-efficiency.
* **Intelligent Chunking:** Implements `RecursiveCharacterTextSplitter` to maintain semantic context.
* **Vector Search:** Utilizes **FAISS** and **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) for fast retrieval.
* **Streamlined UI:** A custom-styled **Streamlit** interface with a focused "Assistant" persona.
* **Streaming Responses:** Real-time text generation for a smoother user experience.

---

## 🛠️ Tech Stack
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **LLM:** [Ollama](https://ollama.com/) (Mistral)
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)
* **Embeddings:** [HuggingFace Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* **Frontend:** [Streamlit](https://streamlit.io/)

---

## 📋 Prerequisites
1.  **Ollama Installed:** Download and install from [ollama.ai](https://ollama.com/).
2.  **Model Pulled:** Run the following command in your terminal:
    ```bash
    ollama pull mistral
    ```
3.  **Python 3.9+**

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/A-SH2k17/SimpleRAG.git