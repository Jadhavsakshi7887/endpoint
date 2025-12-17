# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys
# Project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vectorstore_manager import initialize_vectorstore
from rag_query import rag_query

app = FastAPI(title="RAG Bot API", version="1.0.0")



# Hardcoded document path
HARDCODED_FILE = project_root / r"motia_docs\motia.pdf"
if not HARDCODED_FILE.exists():
    raise FileNotFoundError(f"Hardcoded file not found: {HARDCODED_FILE}")

# Initialize vectorstore at startup
initialize_vectorstore(str(HARDCODED_FILE))

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    success: bool

# Health check
@app.get("/")
async def root():
    return {"message": "RAG Bot API is running", "status": "healthy"}

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    answer = rag_query(request.question)
    
    if not answer:
        raise HTTPException(status_code=500, detail="No answer generated")
    
    return QueryResponse(answer=answer, success=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
