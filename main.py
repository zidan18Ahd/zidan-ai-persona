import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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

@app.post("/vapi/webhook/chat/completions")
@app.post("/chat/completions")
async def vapi_webhook(request: Request):
    try:
        body = await request.json()
        print(f"Incoming Payload: {json.dumps(body)}")
        
        user_query = ""
        
        # 1. Parse Vapi's specific Custom LLM message structure
        if "message" in body and "messages" in body["message"]:
            vapi_messages = body["message"]["messages"]
            if vapi_messages:
                user_query = vapi_messages[-1].get("content", "")
                
        # 2. Fallback to standard OpenAI array format
        elif "messages" in body:
            openai_messages = body.get("messages", [])
            if openai_messages:
                user_query = openai_messages[-1].get("content", "")
                
        print(f"Extracted Query text: '{user_query}'")
        
        # If we couldn't parse anything, use a generic fallback query
        if not user_query:
            user_query = "Hello"

        # Fetch answer from your local RAG architecture engine
        answer = await get_answer(user_query, [])
        print(f"RAG Response text: {answer}")
        
        # A clean asynchronous chunk generator for Vapi's required streaming style
        async def sse_generator():
            chunk_id = "chatcmpl-vapi"
            words = str(answer).split(" ")
            
            for i, word in enumerate(words):
                space = " " if i < len(words) - 1 else ""
                content_chunk = f"{word}{space}"
                
                chunk_data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content_chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.01)  # Lightning fast streaming token delivery
            
            # Send the official closing stop payload packet
            stop_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")
        
    except Exception as e:
        print(f"Fallback crash protection activated: {e}")
        # Secure fallback block response configuration to keep Vapi alive no matter what
        return JSONResponse({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "I am looking into Zidan's project repositories right now."},
                "finish_reason": "stop"
            }]
        })

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