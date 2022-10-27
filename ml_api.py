# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:42:18 2022

@author: Antonio
"""

from fastapi import FastAPI, UploadFile
from prediction import predict
import shutil

app = FastAPI()


    
@app.post('/ml/')
def makePred(audio: UploadFile):
    path = f'.//audios//{audio.filename}'
    with open(path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    return predict(path)