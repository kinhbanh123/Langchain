import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.document_loaders import PyPDFLoader  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
#from langchain.vectorstores import Chroma #vector store
import weaviate #vector store
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import pickle
import re
import csv
from weaviate import classes as wvc
###########################################################################
load_dotenv() #Load something secret
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
if not huggingfacehub_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the enviroment")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the enviroment")
# Set up the Google model
#llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY , streaming=True)
llm = OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.1, streaming=True)
# Set up the HuggingFace model
repo_id = "HuggingFaceH4/zephyr-7b-beta"
"""llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                    repo_id=repo_id,
                   model_kwargs={"temperature":0.01, "max_new_tokens":1000},
                   streaming=True) #swap model if you wanna"""

template = """
<|system|>>
You are an anime girl AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers.
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
# Set these environment variables
URL = "https://langchainlearn-1x0cyj2w.weaviate.network"
APIKEY = "RqZAcXk3do3U6ihpQtDYnNubuyEaktLBWRuu"
# Connect to a WCS instance
client = weaviate.connect_to_wcs(
    cluster_url=URL,
    auth_credentials=weaviate.auth.AuthApiKey(APIKEY))

global embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
output_parser = StrOutputParser()
csv_path = "./data/file_to_id.csv"
global name_data
name_data = "vector_db"
global collection
collection = client.collections.get(name_data)
###########################################################################
def createNewCollection():
    client.collections.delete(name_data)
    # lets make sure its vectorizer is what the one we want
    collection = client.collections.create(
    name=name_data,
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
    generative_config=wvc.config.Configure.Generative.openai(),)
###########################################################################

###########################################################################
###########################################################################
def load_single_pdf(file_path: str):
    if not file_path.endswith(".pdf"):
        raise ValueError("File is not a PDF")
    if not os.path.isfile(file_path):
        raise ValueError("File does not exist")
    ids = []
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    name = file_path.split('/')[-1]

    # Read existing file-to-id mappings
    file_to_id = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 2:
                    file_to_id[row[0]] = row[1]
    
    # Check if file name already exists in CSV
    if name in file_to_id:
        print(f"File {name} already exists in the CSV. Skipping.")
        return pages  # Return the pages without further processing

    doc_search = WeaviateVectorStore.from_documents(pages, embedding=embeddings, client=client,index_name=name_data)
    for item in collection.iterator(include_vector=True):
        uuid_str = str(item.uuid)  # Convert UUID to string for comparison
        if not any(uuid_str in str(file_to_id) for file_to_id in file_to_id.values()):
            ids.append(item.uuid)
    print(ids)
    file_to_id[name] = ids
    with open(csv_path, mode="a", newline='', encoding='utf-8') as csvfile:  # Use "a" mode to append new entries
        writer = csv.writer(csvfile)
        writer.writerow([name, ids])
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
    doc_search = WeaviateVectorStore.from_documents(pages, embedding=embeddings, client=client,index_name=name_data)
    print(f'You have {len(documents)} documents in your data')
    print(f'There are {len(documents[0].page_content)} characters in your document')
    return documents #chua can toi
###########################################################################
def load_split_docs(db_path="./data/"):
    with open(os.path.join(db_path, "split_docs.pkl"), "rb") as f:
        split_docs = pickle.load(f)
    return split_docs #load file split cho setup
###########################################################################
def CreateSplitDocs(db_path="./data/"):
    # Split documents
    docs = LoadDocs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    # Save split_docs using pickle
    with open(os.path.join(db_path, "split_docs.pkl"), "wb") as f:
        pickle.dump(split_docs, f)
###########################################################################
def LoadDocs(db_path="./data/",csv_path = "./data/file_to_id.csv"):
    docs = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                filename = row[0]  # Tên file PDF
                pdf_path = os.path.join(db_path, filename)
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                docs.extend(pages)
    return docs
##########################################################################
def delete_entry_from_csv(file_name, csv_path = "./data/file_to_id.csv"):
    # Đọc dữ liệu từ file CSV vào một từ điển file_to_id
    file_to_id = {}
    with open(csv_path, mode="r", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                file_to_id[row[0]] = row[1]
    # Kiểm tra xem file_name có trong từ điển không
    if file_name in file_to_id:
        # Lấy doc_id tương ứng
        doc_id = file_to_id.pop(file_name)
        print(f"Deleted entry for {file_name} with ID: {doc_id}")
    else:
        print(f"Entry for {file_name} not found in CSV.")
    collection.data.delete_by_id(doc_id)
    # Ghi lại các cặp tên file và ID còn lại vào file CSV
    with open(csv_path, mode="w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for file, ids in file_to_id.items():
            writer.writerow([file, ids])
###########################################################################
def setup_retrieval_chain():
    #docs = load_single_pdf("./data/IRyS.pdf")
    docs = LoadDocs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    vs=WeaviateVectorStore(client=client,index_name=name_data,embedding=embeddings,text_key="text")
    return vs.as_retriever(search_kwargs={"k": 2})
###########################################################################
def loaddata():
    vs = setup_retrieval_chain()
#############################################################################
async def generate_chat_responses(message):
    chain = createChain(message)
    async for chunk in chain.astream(message):
        content = chunk.replace("\n", "<br>")
        yield f"data: {content}\n\n"
#############################################################################
def createChain(question):
    vs = setup_retrieval_chain()
    vs.get_relevant_documents(question)
    keyword_retriever = BM25Retriever.from_documents(load_split_docs()) 
    keyword_retriever.k =  2
    keyword_retriever.get_relevant_documents(question)
    retriever_chain= EnsembleRetriever(retrievers=[vs,keyword_retriever],weights=[0.5, 0.5])

    chain = (
        {"context": retriever_chain, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser )
    return chain
load_single_pdf("./data/asteriskNamirin.pdf")
CreateSplitDocs()
"""createNewCollection()
for item in collection.iterator(include_vector=True):
    print(item.uuid)
load_single_pdf("./data/AZKi.pdf")
for item in collection.iterator(include_vector=True):
    print(item.uuid)
load_single_pdf("./data/AZKi.pdf")
for item in collection.iterator(include_vector=True):
    print(item.uuid)
delete_entry_from_csv("IRyS.pdf", csv_path)
for item in collection.iterator(include_vector=True):
    print(item.uuid)"""
#generate_chat_responses("What is the famous song of Namirin")
"""for item in collection.iterator(include_vector=True):
    print(item.uuid)
delete_entry_from_csv("IRyS.pdf", csv_path)
"""


def get_helpful_answer(question: str) -> str:
    chain = createChain(question)
    answer = chain.invoke(question)  
    
    if answer and isinstance(answer, str):
        last_newline_index = answer.rfind('>\n')
        if last_newline_index != -1:
            return answer[last_newline_index + 1:].strip()
    return answer # show all docs
