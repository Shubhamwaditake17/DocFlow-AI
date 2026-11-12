# DocFlow AI: Multi-Agent Document Automation Pipeline

Turns messy invoice text into validated records. Extractor, Validator and Corrector agents run in a
loop; documents that stay invalid or low-confidence go to a human review queue.

```
text -> ExtractorAgent (LLM JSON) -> clean (Pandas/NumPy) -> ValidatorAgent (Pydantic + rules)
     -> invalid? CorrectorAgent re-prompts with the error messages (max 2 retries)
     -> accepted (stored) | needs_review (queue) -> human fixes via API or Streamlit -> reviewed
```

## Run
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key
uvicorn app.main:app --reload            # sqlite by default; set DATABASE_URL for PostgreSQL
streamlit run dashboard/app.py           # results, metrics, review queue
pytest -q                                # mocked LLM, no network
python -m eval.run_eval                  # 50 labeled synthetic invoices, compares extractor prompts
docker compose up --build                # API + PostgreSQL + dashboard
```
API: `POST /process`, `GET /documents`, `GET /review`, `POST /review/{id}`, `GET /metrics`.

## Design notes
- `BaseAgent` interface; agents are swappable. Pydantic `InvoiceFields` is the contract.
- Corrector receives the validator's error text, so retries are targeted rather than blind.
- Eval set is synthetic (`data/generate_docs.py`, seeded); every 10th document has no usable total and must be routed to review.
- `--dry-run` checks the eval harness with an oracle LLM. It is not a model score.
- Log results and failure cases in `docs/experiments.md`.
