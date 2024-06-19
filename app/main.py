import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
loader = PyPDFLoader("./data/Hololive Production.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)
embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)
retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
# Define FastAPI app
app = FastAPI()

# Define request and response models
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

class ActionResponse(BaseModel):
    message: str

@app.post("/chat/", response_model=Answer)
async def chat(question:str):
    try:
        answer = retrieval_chain.run(question)
        pattern = r"Helpful Answer:\s*(.*)"
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            helpful_answer = match.group(1).strip()

        return Answer(answer=helpful_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Welcome to the Chat API"}
