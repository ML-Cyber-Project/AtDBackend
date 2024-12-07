from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Threat(BaseModel):
    name: str
    description: str


@app.post('/threats/')
async def create_threat(threat: Threat):
    return threat
