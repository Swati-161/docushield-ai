from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routers import documents
from backend.config import OUTPUTS_DIR

app = FastAPI(
    title="DocuShield AI",
    description="Real-time document forgery detection",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS_DIR)),
    name="outputs"
)

@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "DocuShield AI",
        "version": "0.2.0"
    }