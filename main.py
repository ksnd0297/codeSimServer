from fastapi import FastAPI
import efficientNet
from pydantic import BaseModel
from typing import List
from starlette.middleware.cors import CORSMiddleware

from Turtle import drawCode, cleanUp

from pullMain import extract_functions, replace_function_calls

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Item(BaseModel):
    source: List[str]

@app.post("/items/")
async def create_item(item: Item):
    source = item.source

    replace_mainSource = []

    for code in source:
        extracted_functions = extract_functions(code)
        modified_functions  = replace_function_calls(extracted_functions)

        for function in modified_functions:
            if(function['name'] == 'main'):
                replace_mainSource.append(function['definition'])
    
    drawCode(replace_mainSource)
    cleanUp()
    result = efficientNet.execute()

    return result