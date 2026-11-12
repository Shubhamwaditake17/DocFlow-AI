import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Text, select, func
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source = Column(String(200))
    status = Column(String(20), index=True)   # accepted | needs_review | reviewed
    vendor = Column(String(200))
    invoice_number = Column(String(100))
    invoice_date = Column(Date)
    total = Column(Float)
    currency = Column(String(3))
    confidence = Column(Float)
    attempts = Column(Integer)
    reason = Column(Text)
    errors = Column(Text)
    raw_json = Column(Text)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def make_sessionmaker(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _apply(doc, fields):
    for k, v in fields.items():
        setattr(doc, k, datetime.date.fromisoformat(v) if k == "invoice_date" else v)


def save(session, source, text, result):
    doc = Document(source=source, text=text, status=result["status"], confidence=result["confidence"],
                   attempts=result["attempts"], reason=result["reason"],
                   errors=json.dumps(result["errors"]), raw_json=json.dumps(result["raw"], default=str))
    if result["fields"]:
        _apply(doc, result["fields"])
    session.add(doc)
    session.commit()
    return doc


def review_fix(session, doc_id, fields, validator):
    doc = session.get(Document, doc_id)
    if doc is None or doc.status != "needs_review":
        return False, ["document not found or not awaiting review"]
    rec, errors = validator.run(fields)
    if rec is None:
        return False, errors
    _apply(doc, rec.model_dump(mode="json"))
    doc.status, doc.confidence = "reviewed", 1.0
    session.commit()
    return True, []


def list_documents(session, status=None):
    q = select(Document).order_by(Document.id.desc())
    if status:
        q = q.where(Document.status == status)
    return session.scalars(q).all()


def counts(session):
    rows = session.execute(select(Document.status, func.count()).group_by(Document.status)).all()
    return {s: n for s, n in rows}
