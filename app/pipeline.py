import json


class Pipeline:
    """extract -> validate -> (correct with error feedback)* -> accept or route to human review."""

    def __init__(self, extractor, validator, corrector, threshold=0.7, max_corrections=2):
        self.extractor, self.validator, self.corrector = extractor, validator, corrector
        self.threshold, self.max_corrections = threshold, max_corrections

    def process(self, text):
        attempts, errors, raw, rec = 0, [], {}, None
        try:
            raw = self.extractor.run(text)
            rec, errors = self.validator.run(raw)
        except (ValueError, json.JSONDecodeError) as e:
            errors = [f"extractor: {e}"]
        while rec is None and attempts < self.max_corrections:
            attempts += 1
            try:
                raw = self.corrector.run(text, raw, errors)
                rec, errors = self.validator.run(raw)
            except (ValueError, json.JSONDecodeError) as e:
                errors = [f"corrector: {e}"]
        try:
            conf = float(raw.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if rec is None:
            status, reason = "needs_review", "validation failed"
        elif conf < self.threshold:
            status, reason = "needs_review", "low confidence"
        else:
            status, reason = "accepted", ""
        return {"status": status, "reason": reason, "fields": rec.model_dump(mode="json") if rec else None,
                "confidence": conf, "attempts": attempts, "errors": errors, "raw": raw}
