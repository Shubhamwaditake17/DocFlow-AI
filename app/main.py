import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app import db
from app.agents import ExtractorAgent, ValidatorAgent, CorrectorAgent
from app.llm import GroqClient
from app.pipeline import Pipeline

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///docflow.db")
EXTRACTOR_PROMPT = os.getenv("EXTRACTOR_PROMPT", "prompts/extractor_v2.txt")
app = FastAPI(title="DocFlow AI")
_state = {}


def get_sessionmaker():
    if "sm" not in _state:
        _state["sm"] = db.make_sessionmaker(DATABASE_URL)
    return _state["sm"]


def get_pipeline():
    if "pipe" not in _state:
        llm = GroqClient()
        _state["pipe"] = Pipeline(ExtractorAgent(llm, EXTRACTOR_PROMPT), ValidatorAgent(),
                                  CorrectorAgent(llm, "prompts/corrector_v1.txt"))
    return _state["pipe"]


def as_dict(d):
    return {"id": d.id, "source": d.source, "status": d.status, "vendor": d.vendor,
            "invoice_number": d.invoice_number, "invoice_date": str(d.invoice_date) if d.invoice_date else None,
            "total": d.total, "currency": d.currency, "confidence": d.confidence,
            "attempts": d.attempts, "reason": d.reason}


class Doc(BaseModel):
    source: str = "api"
    text: str = Field(min_length=10, max_length=20000)


class Fix(BaseModel):
    vendor: str
    invoice_number: str
    invoice_date: str
    total: float
    currency: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
def process(doc: Doc):
    try:
        result = get_pipeline().process(doc.text)
    except KeyError:
        raise HTTPException(500, "GROQ_API_KEY is not set")
    with get_sessionmaker()() as s:
        return {**as_dict(db.save(s, doc.source, doc.text, result)), "errors": result["errors"]}


@app.get("/documents")
def documents(status: str | None = None):
    with get_sessionmaker()() as s:
        return [as_dict(d) for d in db.list_documents(s, status)]


@app.get("/review")
def review_queue():
    return documents("needs_review")


@app.post("/review/{doc_id}")
def review(doc_id: int, fix: Fix):
    with get_sessionmaker()() as s:
        ok, errors = db.review_fix(s, doc_id, fix.model_dump(), ValidatorAgent())
    if not ok:
        raise HTTPException(422, errors)
    return {"status": "reviewed"}


@app.get("/metrics")
def metrics():
    with get_sessionmaker()() as s:
        c = db.counts(s)
    total = sum(c.values())
    return {"counts": c, "total": total,
            "auto_accept_rate": round(c.get("accepted", 0) / total, 3) if total else None}
