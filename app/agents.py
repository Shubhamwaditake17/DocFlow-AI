import json
import re
from abc import ABC, abstractmethod
from datetime import date
from pydantic import ValidationError
from app.cleaning import clean_record
from app.schemas import InvoiceFields, FIELDS


def parse_llm_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in LLM reply")
    return json.loads(m.group(0))


class BaseAgent(ABC):
    name = ""

    @abstractmethod
    def run(self, *args, **kwargs): ...


class ExtractorAgent(BaseAgent):
    name = "extractor"

    def __init__(self, llm, prompt_path):
        self.llm, self.system = llm, open(prompt_path).read()

    def run(self, text):
        return parse_llm_json(self.llm.complete(self.system, text))


class ValidatorAgent(BaseAgent):
    name = "validator"

    def run(self, raw):
        cleaned = clean_record(raw)
        try:
            rec = InvoiceFields(**{k: cleaned.get(k) for k in FIELDS})
        except ValidationError as e:
            return None, [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()]
        if rec.invoice_date > date.today():
            return None, ["invoice_date: date is in the future"]
        return rec, []


class CorrectorAgent(BaseAgent):
    name = "corrector"

    def __init__(self, llm, prompt_path):
        self.llm, self.system = llm, open(prompt_path).read()

    def run(self, text, previous, errors):
        user = (f"DOCUMENT:\n{text}\n\nPREVIOUS JSON:\n{json.dumps(previous)}\n\n"
                "ERRORS:\n- " + "\n- ".join(errors))
        return parse_llm_json(self.llm.complete(self.system, user))
