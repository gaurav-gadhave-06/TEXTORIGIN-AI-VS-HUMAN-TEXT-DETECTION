from pathlib import Path

# Define the content of the README.md
readme_content = """# 🧠 TextOrigin: AI vs Human Text Classifier

TextOrigin is a powerful and intuitive **text origin detection system** designed to classify whether a piece of text was **AI-generated** or **human-written**. By combining cutting-edge technologies like **Fine-tuning**, **Retrieval-Augmented Generation (RAG)**, and **Pinecone**, it ensures accurate, context-aware classifications.

---

## 🚀 Features

- 🔍 **Fine-Tuned Classification**: Customized model trained specifically to detect AI vs human-written content.
- 📚 **RAG-Powered Contextual Reasoning**: Retrieval-Augmented Generation enhances decision-making by including external context.
- ⚡ **Fast, Scalable Retrieval**: Powered by Pinecone for lightning-fast, scalable document search.
- 📂 **Multi-Format Uploads**: Analyze `.txt`, `.pdf`, and `.docx` files seamlessly.
- 🧾 **Detailed Explanations**: Get confidence scores and reasoning behind predictions.
- 🖥️ **Interactive Streamlit Interface**: A user-friendly web app for hands-on exploration.

---

## 📌 Why Use This System?

### ✅ Fine-Tuning
Fine-tuning adapts a pre-trained model to the **specific task** of distinguishing between human-written and AI-generated text, improving:

- 🎯 Accuracy on domain-specific samples
- 🤖 Custom classification labels with confidence and explanations

> Pre-trained models are general-purpose. Fine-tuning makes them **task-specialized and more precise**.

✅ What is DistilBERT?
DistilBERT is a smaller, faster, and lighter version of BERT (Bidirectional Encoder Representations from Transformers), developed using knowledge distillation while retaining 97% of BERT’s performance on classification tasks.

🤔 Why Not Use Other Models?
Model	Reason Not Used
BERT	More accurate but computationally expensive and slower for inference.
RoBERTa / DeBERTa	More powerful but much heavier and not necessary for binary classification.
GPT-style models	Primarily generative, not classification-focused, and require more computation and prompt-engineering for prediction.
LLaMA or LLMs	Better for RAG or generation; not ideal for fast, focused classification tasks.

🧩 Conclusion: For a lightweight, high-performance, and real-time classifier, DistilBERT hits the sweet spot.

---

### 🧠 RAG (Retrieval-Augmented Generation)
RAG enhances context by **retrieving relevant external documents** before generating a response. It allows:

- 🌐 **Contextual Awareness**: Pulls supporting documents from Pinecone
- 🧠 **Informed Reasoning**: Combines retrieval + generation for smarter detection
- 📈 **Improved Accuracy**: Reduces hallucination and misclassification

---

### 📦 Pinecone for Retrieval
Pinecone stores and retrieves **vector embeddings** used in the RAG pipeline.

- ⚡ **Fast similarity search** with cosine distance
- 🌍 **Scalability** for large datasets
- 🔄 **Real-time retrieval** ensures relevant, fresh context


✅ What is Pinecone?
Pinecone is a managed vector database service built specifically for real-time, high-scale vector similarity search. In this project, it supports RAG (Retrieval-Augmented Generation) by retrieving the most relevant context.


💡 Why Choose Pinecone for RAG?
Reason	Description
⚡ Scalability	Built to handle millions of vectors and scales automatically — perfect for future growth.
🚀 Low Latency	Optimized for real-time similarity search with milliseconds response time.
🔌 Seamless Integration	Works perfectly with LangChain and popular embedding models, which simplifies development.
💾 Persistent & Managed	You don't have to worry about hosting or storage. It's managed infrastructure with high availability.
🔒 Security	Offers built-in security, API key management, and compliance for enterprise use.


> Alternatives like FAISS work, but Pinecone provides **speed, scalability, and seamless LangChain integration**.

---

## 🧱 Project Structure

### 📁 Core Components

| Module              | Description |
|---------------------|-------------|
| `main.py`           | Streamlit interface for the classifier |
| `finetuned_tool.py` | Uses the fine-tuned model for classification |
| `rag_tool.py`       | Uses RAG + Pinecone for context-aware detection |
| `utils.py`          | Text extraction, document handling, and helpers |
| `.env`              | Stores API keys for Groq and Pinecone |

---

## 🧪 How It Works

### 1. 🔐 Environment Setup
```python
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

✅ What is Groq?
Groq provides ultra-fast inference for large language models (LLMs) using custom-built Language Processing Units (LPUs). It supports models like LLaMA 3 with single-digit millisecond latency, making it ideal for real-time applications.

🧠 Why Choose Groq?
Feature	Benefit for This Project
⚡ Speed (ms-level latency)	Groq offers blazing fast inference (often <10ms) compared to traditional GPU-based systems. Great for real-time response in web apps.
🧠 Supports Large LLMs (LLaMA 3)	Enables you to use cutting-edge, large-scale LLMs like LLaMA 3–70B, ideal for complex context generation and text reasoning.
📈 Deterministic Output	With temperature=0.0, Groq returns consistent and repeatable responses — important for classification-style use cases.
🛠️ API-Based Simplicity	Works as a hosted API with high availability — you don’t have to manage local models, memory, or compute infrastructure.
🤝 LangChain Support	Groq integrates well with LangChain, making it easy to plug into RAG and tool-based agents.


📊 Comparison Table
Feature	Groq	Ollama
Inference Speed	✅ Ultra-fast (ms)	❌ Slower (sec)
LLM Size	✅ LLaMA 3–70B	❌ Up to 7B (mostly)
Hosting	✅ Cloud-based, scalable	❌ Local only
Deployment	✅ API & LangChain Ready	⚠️ Dev/test use, not ideal for large-scale prod
Real-time Use	✅ Excellent	⚠️ Good for small loads
Cost	✅ Pay-per-use (no infra needed)	✅ Free if local compute is available
