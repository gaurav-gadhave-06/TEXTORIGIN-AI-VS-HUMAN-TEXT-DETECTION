import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

def get_rag_chain(index_name="human-vs-ai"):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

    llm = ChatGroq(model='llama3-70b-8192', temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def rag_detect(text, rag_chain=None):
    if rag_chain is None:
        rag_chain = get_rag_chain()
    response = rag_chain.invoke({"input": text})
    return response["answer"]