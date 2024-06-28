import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.document_loaders import PyPDFLoader  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.vectorstores import Chroma #vector store
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter  

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import pickle
import re
import csv
###########################################################################
load_dotenv() #Load something secret
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not huggingfacehub_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the enviroment")
# Set up the Google model
#llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY )
# Set up the HuggingFace model
repo_id = "HuggingFaceH4/zephyr-7b-beta"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.01, "max_new_tokens":1000} ) #swap model if you wanna
"""llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY)""" #swap model if you wanna
template = """
<|system|>>
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT
Keep in mind, you will lose the job, if you answer out of CONTEXT questions
CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
###########################################################################
prompt = ChatPromptTemplate.from_template(template)
global embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
output_parser = StrOutputParser()
csv_path = "./data/file_to_id.csv"

###########################################################################
def load_single_pdf(file_path: str):
    if not file_path.endswith(".pdf"):
        raise ValueError("File is not a PDF")
    if not os.path.isfile(file_path):
        raise ValueError("File does not exist")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    doc_search = Chroma.from_documents(pages, embeddings)
    document_ids = doc_search.get()["ids"]
    document_ids = doc_search.get()["ids"]
    print("Document IDs:", document_ids)
    # Convert the document IDs to a single string with comma-separated values
    ids_str = ",".join(document_ids)
    # Read the existing file-to-id mappings
    file_to_id = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 2:
                    file_to_id[row[0]] = row[1]
    if file_path in file_to_id:
        print(f"File {file_path} already exists in the CSV. Overwriting entry.")
    file_to_id[file_path] = ids_str
    # Write the updated mappings back to the CSV file
    with open(csv_path, mode="w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for file, ids in file_to_id.items():
            writer.writerow([file, ids])
    return pages
###########################################################################
#Load all pdf
def load_all_pdfs_from_folder(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            documents.extend(pages)
    return documents #tra ve thong tin cua nhieu trang lay ten va id vao mot pdf
###########################################################################
def load_split_docs(db_path="./data/"):
    with open(os.path.join(db_path, "split_docs.pkl"), "rb") as f:
        split_docs = pickle.load(f)
    return split_docs #load file split cho setup
###########################################################################
def add_documents_to_chroma(docs, db_path="./data/"):
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    # Save split_docs using pickle
    with open(os.path.join(db_path, "split_docs.pkl"), "wb") as f:
        pickle.dump(split_docs, f)
    # Create embeddings
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    doc_search = Chroma.from_documents(split_docs, embeddings,persist_directory="./data/chroma_db")
    doc_search.add_documents(split_docs)
"""    # Initialize or load existing Chroma DB
    if os.path.exists(db_path):
        try:
            doc_search = Chroma.load(db_path, embeddings)
        except:
            doc_search = Chroma.from_documents(split_docs, embeddings)
    else:"""
    
###########################################################################
def delete_entry(file_path: str):
    # Đọc nội dung hiện tại của file-to-id CSV
    file_to_id = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 2:
                    file_to_id[row[0]] = row[1]
    # Kiểm tra xem file_path có trong từ điển không
    if file_path in file_to_id:
        # Lấy danh sách document IDs
        document_ids = file_to_id[file_path].split(',')

        # Tải Chroma DB
        doc_search = load_chroma_db()

        # Xóa các document IDs khỏi Chroma DB
        for doc_id in document_ids:
            doc_search.delete([doc_id])
        doc_search(persist_directory="./data/chroma_db", embedding_function=embeddings)
        # Xóa mục từ từ điển
        del file_to_id[file_path]
        print(f"Entry for {file_path} deleted from CSV and Chroma DB.")
    else:
        print(f"Entry for {file_path} not found in CSV.")

    # Ghi lại các ánh xạ đã cập nhật vào file CSV
    with open(csv_path, mode="w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for file, ids in file_to_id.items():
            writer.writerow([file, ids])

    print(f"File {csv_path} was successfully updated.")
###########################################################################
# Tải ChromaDB
def load_chroma_db():
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)

###########################################################################
def setup_retrieval_chain():
    #folder_path = "./data"
    #docs = load_all_pdfs_from_folder(folder_path)
    #splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    #split_docs = splitter.split_documents(docs)
    """embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    doc_search = Chroma.from_documents(split_docs, embeddings)"""
    db_path = "./data/vectordb"
    doc_search = load_chroma_db()
    #return RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
    retriever_vectordb = doc_search.as_retriever(search_kwargs={"k": 2})     
    keyword_retriever = BM25Retriever.from_documents(load_split_docs())
    keyword_retriever.k =  2
    return  EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])
"""loader = PyPDFLoader("./data/Hololive Production.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)
embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)"""

#docs = load_all_pdfs_from_folder("./data")
#docs = load_single_pdf("./data/IRyS.pdf")
#delete_entry("./data/AZKi.pdf")
#add_documents_to_chroma(docs)
retrieval_chain = setup_retrieval_chain()

def loaddata():
    retrieval_chain = setup_retrieval_chain()

chain = (
    {"context": retrieval_chain, "query": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser 
)
    
def get_helpful_answer(question: str) -> str:
    
    answer = chain.invoke(question)  
    
    if answer and isinstance(answer, str):
        last_newline_index = answer.rfind('>\n')
        if last_newline_index != -1:
            return answer[last_newline_index + 1:].strip()
    return answer # show all docs


question = "Write lyrics of the song name Love Story by Taylor Swift"
for chunk in chain.stream(question):
    print(chunk)