"""Real run: python -m eval.run_eval   (needs GROQ_API_KEY)
Harness check: python -m eval.run_eval --dry-run   (oracle LLM, NOT model accuracy)"""
import argparse, datetime, glob, json, os, sys, time
from app.agents import ExtractorAgent, ValidatorAgent, CorrectorAgent
from app.llm import LLMClient
from app.pipeline import Pipeline
from app.schemas import FIELDS
from data.generate_docs import generate


class OracleLLM(LLMClient):
    """Returns gold fields; every 5th doc gets a bad date first so the corrector path is exercised."""

    def __init__(self, docs):
        self.by_text = {d["text"]: d for d in docs}

    def _fields(self, doc, broken):
        g = doc["gold"] or {"vendor": "Unknown Co", "invoice_number": "X-1", "invoice_date": "2025-01-01",
                            "total": "TBC", "currency": "USD"}
        out = {**g, "confidence": 0.95 if doc["gold"] else 0.3}
        if broken:
            out["invoice_date"] = "31/31/2025"
        return json.dumps(out)

    def complete(self, system, user):
        if system.startswith("You correct"):
            text = user.split("DOCUMENT:\n", 1)[1].split("\n\nPREVIOUS JSON:", 1)[0]
            return self._fields(self.by_text[text], broken=False)
        doc = self.by_text[user]
        return self._fields(doc, broken=doc["id"] % 5 == 0 and doc["gold"] is not None)


def same(field, got, want):
    return abs(float(got) - want) < 0.01 if field == "total" else str(got) == str(want)


def evaluate(pipeline, docs):
    ok_fields = acc_fields = acc_ok = accepted = complete = caught = incomplete = attempts = 0
    lat = []
    for d in docs:
        t = time.perf_counter()
        r = pipeline.process(d["text"])
        lat.append((time.perf_counter() - t) * 1000)
        attempts += r["attempts"]
        if d["gold"] is None:
            incomplete += 1
            caught += r["status"] == "needs_review"
            continue
        complete += 1
        if r["status"] == "accepted":
            accepted += 1
            hits = sum(same(f, r["fields"][f], d["gold"][f]) for f in FIELDS)
            ok_fields, acc_fields, acc_ok = ok_fields + hits, acc_fields + len(FIELDS), acc_ok + hits
    n = len(docs)
    return {"n": n, "field_accuracy": ok_fields / max(complete * len(FIELDS), 1),
            "accepted_precision": acc_ok / max(acc_fields, 1), "auto_rate": accepted / max(complete, 1),
            "review_catch_rate": caught / max(incomplete, 1), "avg_attempts": attempts / n,
            "avg_ms": sum(lat) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="docs/experiments.md")
    args = ap.parse_args()
    if not args.dry_run and "GROQ_API_KEY" not in os.environ:
        sys.exit("Set GROQ_API_KEY first (or use --dry-run to test the harness).")
    docs = generate()
    lines = [f"\n## Run {datetime.datetime.now():%Y-%m-%d %H:%M}" + (" (DRY RUN: harness check only)" if args.dry_run else ""),
             "| Extractor prompt | Docs | Field accuracy | Precision on accepted | Auto-accept | Review catch | Avg corrections | Avg ms |",
             "|---|---|---|---|---|---|---|---|"]
    for prompt in sorted(glob.glob("prompts/extractor_v*.txt")):
        if args.dry_run:
            llm = OracleLLM(docs)
        else:
            from app.llm import GroqClient
            llm = GroqClient()
        pipe = Pipeline(ExtractorAgent(llm, prompt), ValidatorAgent(), CorrectorAgent(llm, "prompts/corrector_v1.txt"))
        r = evaluate(pipe, docs)
        lines.append(f"| {os.path.basename(prompt)} | {r['n']} | {r['field_accuracy']:.0%} | {r['accepted_precision']:.0%} | "
                     f"{r['auto_rate']:.0%} | {r['review_catch_rate']:.0%} | {r['avg_attempts']:.2f} | {r['avg_ms']:.0f} |")
    print("\n".join(lines))
    open(args.out, "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
