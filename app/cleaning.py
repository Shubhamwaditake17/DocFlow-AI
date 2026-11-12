"""Pandas/NumPy normalization of raw LLM output before validation."""
import re
import numpy as np
import pandas as pd

CURRENCY = {"$": "USD", "US$": "USD", "USD": "USD", "€": "EUR", "EUR": "EUR",
            "₹": "INR", "RS": "INR", "INR": "INR", "£": "GBP", "GBP": "GBP"}


def clean_amount(v):
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^\d.,-]", "", str(v))
    if not re.search(r"\d", s):
        return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if re.search(r",\d{1,2}$", s) else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_currency(v):
    key = str(v).strip().upper()
    return CURRENCY.get(key, key)


def clean_date(v):
    ts = pd.to_datetime(str(v), errors="coerce")
    return None if pd.isna(ts) else ts.date().isoformat()


def clean_record(raw):
    out = {k: (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
    if "total" in out:
        amount = clean_amount(out["total"])
        out["total"] = None if np.isnan(amount) else amount
    if "currency" in out:
        out["currency"] = clean_currency(out["currency"])
    if "invoice_date" in out:
        out["invoice_date"] = clean_date(out["invoice_date"])
    return out


def clean_frame(df):
    return pd.DataFrame([clean_record(r) for r in df.to_dict("records")])
