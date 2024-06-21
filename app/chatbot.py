import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.document_loaders import PyPDFLoader  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma #vector store
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from dotenv import load_dotenv
import re

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
                     model_kwargs={"temperature":0.2, "max_new_tokens":1000})

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
"""prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    doc_search = Chroma.from_documents(split_docs, embeddings)
    #return RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
    retriever_vectordb = doc_search.as_retriever(search_kwargs={"k": 2})
    keyword_retriever = BM25Retriever.from_documents(split_docs)
    keyword_retriever.k =  2
    return  EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])
"""loader = PyPDFLoader("./data/Hololive Production.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)
embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)"""

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
    return answer