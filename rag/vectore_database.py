from langchain_community.document_loaders import WebBaseLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "human-vs-ai"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
)


def WebsiteLoader(urls):
    loader = WebBaseLoader(urls)
    return loader.load()


urls = [
        "https://surferseo.com/blog/detect-ai-content/#:~:text=1.,%2Dknown%20over%2Doptimization%20issues.&text=Take%20the%20example%20of%20this,the%20same%20idea%20more%20concisely.",
        "https://medium.com/@raja.gupta20/how-to-detect-texts-generated-by-chatgpt-answered-by-chatgpt-itself-a190146682a9",
        "https://medium.com/@SandleenShah/how-to-detect-ai-vs-human-written-content-a-comparison-f3a13903dca3",
        "https://medium.com/digital-miru/detecting-ai-generated-text-how-it-works-and-why-it-matters-77477f386f97"]

website_docs = WebsiteLoader(urls)


def load_csvs(file_path):
    docs = []
    # docs.extend(CSVLoader(file_path=file_path).load())
    # return docs
    df = pd.read_csv(file_path, nrows=10000)
    documents = [Document(page_content=str(row)) for row in df.astype(str).values.tolist()]
    docs.extend(documents)
    return docs

csv_docs = load_csvs(r'D:\AI & ML\Projects\TextOrigin AI vs Human Text Detection Examples\data\filtered_dataset.csv')



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,chunk_overlap=100
)

all_documents = website_docs + csv_docs

splited_documents = text_splitter.split_documents(all_documents)

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=splited_documents,
    index_name=index_name,
    embedding=embeddings, 
)

print("data inserted in pinecone.")