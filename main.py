import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from rag.ingest import build_vectorstore
from rag.retriever import get_answer
from vapi.handler import handle_webhook

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize or load the ChromaDB vector store on application startup
    print("Building / loading vector store...")
    build_vectorstore()
    print("Ready.")
    yield

app = FastAPI(title="Zidan Ahmed - AI Persona", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

@app.get("/")
async def serve_ui():
    return FileResponse("frontend/index.html")

@app.post("/chat")
async def chat(req: ChatRequest):
    # Handles RAG-grounded queries arriving from the web client
    try:
        answer = await get_answer(req.message, req.history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    # Core webhook listener tracking live events and tool routing from Vapi voice calls
    body = await request.json()
    result = await handle_webhook(body)
    return JSONResponse(result)

@app.get("/health")
async def health():
    return {"status": "ok", "persona": "Zidan Ahmed AI", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "prod") == "dev",
    )