import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.document_loaders import PyPDFLoader  
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma #vector store
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from dotenv import load_dotenv
import re

load_dotenv() #Load something secret
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingfacehub_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment")

# Set up the HuggingFace model
repo_id = "HuggingFaceH4/zephyr-7b-beta"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":1000})

template = """Question: {question}
Answer: Let's give a detailed answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)
#Load all pdf
def load_all_pdfs_from_folder(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            documents.extend(pages)
    return documents

def setup_retrieval_chain():
    folder_path = "./data"
    docs = load_all_pdfs_from_folder(folder_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings()
    doc_search = Chroma.from_documents(split_docs, embeddings)
    return RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())

"""loader = PyPDFLoader("./data/Hololive Production.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)
embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)"""



retrieval_chain = setup_retrieval_chain()

def loaddata():
    retrieval_chain = setup_retrieval_chain()
    
def get_helpful_answer(question: str) -> str:
    answer = retrieval_chain.run(question)
    pattern = r"Helpful Answer:\s*(.*)"
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    return answer