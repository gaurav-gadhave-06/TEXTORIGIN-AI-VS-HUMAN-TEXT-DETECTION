import tempfile
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from rag.rag_detector import rag_detect
from model_finetuning.model import predict


load_dotenv()
# Load the environment variables from the .env file
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# Ensure the environment variables are set
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model='llama3-70b-8192', temperature=0.0)

def finetuned_tool(text):
    label, confidence, reason = predict(text)
    return f"Fine-tuned model: {label} (confidence: {confidence:.2f}). Reason: {reason}"

def rag_tool(text):
    answer = rag_detect(text)
    return f"RAG model: {answer}"

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
                    finetuned_result = finetuned_tool(text)
                    user_query = text.strip()
                    # 2. Pass both to RAG (rag_detect should be updated to accept both)
                    # If your rag_detect only accepts one argument, combine them:
                    rag_input = f"User Query:\n{user_query}\n\nFine-tuned Classifier Output:\n{finetuned_result}"
                    rag_final_response = rag_tool(rag_input)
                    st.markdown("### 🔍 Detection Result")
                    st.write(rag_final_response)
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")