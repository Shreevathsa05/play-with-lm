from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
from pydantic import BaseModel
from src.core.config import settings

class TextToImageRequest(BaseModel):
    model_id:str
    prompt:str
    height:int=512
    width:int=512
    num_images:int=1
    steps:int=50
    guidance_scale:float=7.5

app=FastAPI()

class Tea(BaseModel):
    id: int
    name: str
    origin: str

teas:List[Tea]=[]

@app.get("/")
def read_root():
    return {"Hello": "World"}

# create tea crud

@app.post("/tea")
def save_tea(tea: Tea):
    teas.append(tea)
    return tea

