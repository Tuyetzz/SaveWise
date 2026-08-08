import requests
from fastapi import FastAPI

app = FastAPI(title="SaveWise Brain")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
