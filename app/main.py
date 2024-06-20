from fastapi import FastAPI, HTTPException, UploadFile, File
from chatbot import get_helpful_answer, loaddata
from crawldata import crawl
from pydantic import BaseModel
import os
import shutil

loaddata()
# Define FastAPI app
app = FastAPI()

# Define request and response models
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.post("/chat/", response_model=Answer)
async def chat(question:str):
    try:
        helpful_answer = get_helpful_answer(question)
        return Answer(answer=helpful_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/crawl/")
async def handle_crawl(link: str):
    try:
        file_path = crawl(link)
        return {"message": "Crawling completed", "file_path": file_path}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_location = f"./data/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/")
async def delete_pdf(filename: str):
    file_path = f"./data/{filename}"
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"info": f"file '{filename}' deleted"}
    else:
        raise HTTPException(status_code=404, detail="File not found")