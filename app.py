import mimetypes
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from service import SummarizerService

# Fix Windows MIME-type issue for CSS and JS files
mimetypes.add_type("text/css", ".css", True)
mimetypes.add_type("application/javascript", ".js", True)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Custom Transformer Summarizer",
    description="Pre-Norm RMSNorm + SwiGLU Encoder-Decoder Summarizer",
    version="2.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
summarizer = SummarizerService()


class SummarizeRequest(BaseModel):
    dialogue: str = Field(..., min_length=10, description="Input text or dialogue to summarize")
    max_len: int = Field(default=80, ge=20, le=150)
    beam_size: int = Field(default=4, ge=1, le=6)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/summarize/")
async def summarize(payload: SummarizeRequest):
    try:
        return summarizer.summarize(
            text=payload.dialogue,
            max_len=payload.max_len,
            beam_size=payload.beam_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(exc)}")