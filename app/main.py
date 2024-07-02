from fastapi import FastAPI, HTTPException, UploadFile, File
from chatbot import  loaddata, load_single_pdf, generate_chat_responses, delete_entry_from_db
from fastapi.responses import StreamingResponse, FileResponse
from crawldata import crawl
from pydantic import BaseModel
import os
import shutil
from typing import List

loaddata()
# Define FastAPI app
app = FastAPI()

# Define request and response models
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str


@app.post("/crawl/")
async def handle_crawl(link: str):
    try:
        file_path = crawl(link)
        load_single_pdf(file_path)
        return {"message": "Crawling completed", "file_path": file_path}
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_location = f"./data/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        load_single_pdf(file_location)
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.delete("/delete/")
async def delete_pdf(filename: str):
    file_path = f"./data/{filename}"
    
    if os.path.exists(file_path):
        delete_entry_from_db(filename)
        os.remove(file_path)
        return {"info": f"file '{filename}' deleted"}
    else:
        raise HTTPException(status_code=404, detail="File not found")
@app.get("/list-files/", response_model=List[str])
async def list_files():
    folder_path = "./data"
    try:
        files = os.listdir(folder_path)
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/chat_stream/{message}")
async def chat_stream(message: str):
    return StreamingResponse(generate_chat_responses(message=message), media_type="text/event-stream")
