import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingfacehub_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment")

# Set up the HuggingFace model
repo_id = "HuggingFaceH4/zephyr-7b-beta"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})

template = """Question: {question}
Answer: Let's give a detailed answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)

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
        answer = chain.run(question)
        return Answer(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Welcome to the Chat API"}
