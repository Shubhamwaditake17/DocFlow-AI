import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import db, main
from app.agents import ExtractorAgent, ValidatorAgent, CorrectorAgent
from app.cleaning import clean_amount, clean_currency, clean_date, clean_record, clean_frame
from app.llm import MockLLM
from app.pipeline import Pipeline

GOOD = {"vendor": "Acme Supplies Ltd", "invoice_number": "INV-1234", "invoice_date": "2025-03-12",
        "total": 1234.5, "currency": "USD", "confidence": 0.9}


def pipe(responses, **kw):
    llm = MockLLM(responses)
    return Pipeline(ExtractorAgent(llm, "prompts/extractor_v2.txt"), ValidatorAgent(),
                    CorrectorAgent(llm, "prompts/corrector_v1.txt"), **kw), llm


def test_cleaning_functions():
    assert clean_amount("$1,234.50") == 1234.5
    assert clean_amount("1.234,50") == 1234.5
    assert np.isnan(clean_amount("to be confirmed"))
    assert clean_currency("€") == "EUR" and clean_currency("usd") == "USD"
    assert clean_date("12 Mar 2025") == "2025-03-12" and clean_date("March 12, 2025") == "2025-03-12"
    assert clean_date("31/31/2025") is None
    assert clean_record({"total": "₹ 500", "currency": "₹"})["total"] == 500.0
    assert clean_frame(__import__("pandas").DataFrame([{"total": "$2,000"}])).loc[0, "total"] == 2000.0


def test_validator_accepts_and_rejects():
    v = ValidatorAgent()
    rec, errs = v.run(GOOD)
    assert rec and not errs
    rec, errs = v.run({**GOOD, "total": None, "invoice_date": "2999-01-01"})
    assert rec is None and errs
    assert v.run({**GOOD, "currency": "XYZ"})[0] is None


def test_accepts_clean_extraction():
    p, llm = pipe([json.dumps(GOOD)])
    r = p.process("doc")
    assert r["status"] == "accepted" and r["attempts"] == 0 and len(llm.calls) == 1


def test_corrector_fixes_bad_output_with_error_feedback():
    bad = {**GOOD, "invoice_date": "not a date"}
    p, llm = pipe([json.dumps(bad), json.dumps(GOOD)])
    r = p.process("doc")
    assert r["status"] == "accepted" and r["attempts"] == 1
    assert "invoice_date" in llm.calls[1][1]          # error text was fed back


def test_persistent_failure_goes_to_review_after_max_corrections():
    bad = {**GOOD, "total": None}
    p, _ = pipe([json.dumps(bad)] * 3, max_corrections=2)
    r = p.process("doc")
    assert r["status"] == "needs_review" and r["attempts"] == 2 and r["reason"] == "validation failed"


def test_low_confidence_goes_to_review():
    p, _ = pipe([json.dumps({**GOOD, "confidence": 0.4})])
    r = p.process("doc")
    assert r["status"] == "needs_review" and r["reason"] == "low confidence" and r["fields"]


def test_invalid_json_from_extractor_is_handled():
    p, _ = pipe(["no json here", json.dumps(GOOD)])
    assert p.process("doc")["status"] == "accepted"


@pytest.fixture
def sm(tmp_path):
    return db.make_sessionmaker(f"sqlite:///{tmp_path}/t.db")


def test_db_save_review_and_counts(sm):
    p, _ = pipe([json.dumps({**GOOD, "confidence": 0.2})])
    result = p.process("doc")
    with sm() as s:
        doc = db.save(s, "email", "doc", result)
        assert db.counts(s) == {"needs_review": 1}
        ok, _ = db.review_fix(s, doc.id, {k: GOOD[k] for k in GOOD if k != "confidence"}, ValidatorAgent())
        assert ok and db.counts(s) == {"reviewed": 1}
        assert db.review_fix(s, doc.id, GOOD, ValidatorAgent())[0] is False   # already reviewed


def test_api_process_review_flow(sm, monkeypatch):
    p, _ = pipe([json.dumps({**GOOD, "confidence": 0.1})])
    monkeypatch.setattr(main, "get_pipeline", lambda: p)
    monkeypatch.setattr(main, "get_sessionmaker", lambda: sm)
    c = TestClient(main.app)
    r = c.post("/process", json={"source": "t", "text": "invoice text here"}).json()
    assert r["status"] == "needs_review"
    assert len(c.get("/review").json()) == 1
    fix = {k: GOOD[k] for k in ["vendor", "invoice_number", "invoice_date", "total", "currency"]}
    assert c.post(f"/review/{r['id']}", json=fix).json() == {"status": "reviewed"}
    assert c.post(f"/review/{r['id']}", json=fix).status_code == 422
    assert c.get("/metrics").json()["counts"] == {"reviewed": 1}
    assert c.post("/process", json={"text": "short"}).status_code == 422


def test_dataset_is_deterministic_and_labeled():
    from data.generate_docs import generate
    a, b = generate(), generate()
    assert a == b and len(a) == 50
    assert sum(d["gold"] is None for d in a) == 5


def test_eval_harness_with_oracle():
    from data.generate_docs import generate
    from eval.run_eval import OracleLLM, evaluate
    docs = generate()
    llm = OracleLLM(docs)
    pl = Pipeline(ExtractorAgent(llm, "prompts/extractor_v2.txt"), ValidatorAgent(),
                  CorrectorAgent(llm, "prompts/corrector_v1.txt"))
    r = evaluate(pl, docs)
    assert r["field_accuracy"] == 1.0 and r["review_catch_rate"] == 1.0 and r["avg_attempts"] > 0
