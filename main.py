from fastapi import FastAPI
import efficientNet

app = FastAPI()

@app.get("/")
def 이름():
  efficientNet.execute()
  return "완료"