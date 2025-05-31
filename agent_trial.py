import tempfile
from dotenv import load_dotenv
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from src.prompt import system_prompt
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from rag.rag_detector import rag_detect
from model_finetuning.model import predict
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from src.prompt import agent_prompt_template
import json
from langchain_openai import ChatOpenAI


load_dotenv()
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


llm = ChatGroq(model='llama3-70b-8192', temperature=0.0)

def finetuned_tool(text):
    label, confidence, reason = predict(text)
    return f"Fine-tuned model: {label} (confidence: {confidence:.2f}). Reason: {reason}"

def rag_tool(text):
    answer = rag_detect(text)
    return f"RAG model: {answer}"

tools = [
    Tool(
        name="FineTunedClassifier",
        func=finetuned_tool,
        description="Classifies text as AI or Human using the fine-tuned classifier."
    ),
    Tool(
        name="RAGClassifier",
        func=rag_tool,
        description="Classifies text as AI or Human using the RAG LLM pipeline. Returns a structured analysis with origin, rationale, scores, and suggestions."
    ),
]

# Initialize your LLM (already done as `llm`)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="Text Origin Detector", layout="centered")
st.title("🧠 TextOrigin: AI vs Human Text Classifier")

with st.form("origin_form"):
    user_input = st.text_area("Enter your text and context data:", height=200, placeholder="""Example:
- Input Text: “”
- Context Data: "".
- Reference Data: "".
- User Query: “Is this post AI-generated or human-written?”
""")
    uploaded_file = st.file_uploader("Or upload a document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
    submitted = st.form_submit_button("Analyze")

    if submitted:
        text = user_input

        # Load and extract text from uploaded file (if any)
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif uploaded_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                else:
                    st.error("Unsupported file format")
                    st.stop()

                docs = loader.load()
                file_text = "\n".join([doc.page_content for doc in docs])
                text += "\n\n" + file_text

            except Exception as e:
                st.error(f"File processing failed: {e}")
                st.stop()

        # Analyze the final combined text
        if text.strip():
            with st.spinner("Analyzing text origin..."):
                try:
                    agent_prompt = agent_prompt_template.format(text=text)
                    agent_response = agent.run(agent_prompt)
                    print("RAW AGENT RESPONSE:", repr(agent_response))
                    st.markdown("### 🔍 Detection Result")
                    st.write(agent_response)
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")