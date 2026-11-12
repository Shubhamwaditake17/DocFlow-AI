import os
import pandas as pd
import streamlit as st
from app import db
from app.agents import ValidatorAgent

sm = db.make_sessionmaker(os.getenv("DATABASE_URL", "sqlite:///docflow.db"))
st.title("DocFlow AI")
with sm() as s:
    counts, docs = db.counts(s), db.list_documents(s)
cols = st.columns(3)
for col, key in zip(cols, ["accepted", "needs_review", "reviewed"]):
    col.metric(key.replace("_", " ").title(), counts.get(key, 0))
df = pd.DataFrame([{c: getattr(d, c) for c in ["id", "source", "status", "vendor", "invoice_number",
                    "invoice_date", "total", "currency", "confidence", "attempts", "reason"]} for d in docs])
st.dataframe(df, use_container_width=True)
st.subheader("Review queue")
with sm() as s:
    queue = db.list_documents(s, "needs_review")
for d in queue:
    with st.expander(f"#{d.id} {d.source}: {d.reason}"):
        st.code(d.text)
        with st.form(f"fix{d.id}"):
            f = {"vendor": st.text_input("Vendor", d.vendor or ""),
                 "invoice_number": st.text_input("Invoice number", d.invoice_number or ""),
                 "invoice_date": st.text_input("Date (YYYY-MM-DD)", str(d.invoice_date or "")),
                 "total": st.number_input("Total", value=float(d.total or 0.0)),
                 "currency": st.selectbox("Currency", ["USD", "EUR", "INR", "GBP"])}
            if st.form_submit_button("Approve"):
                with sm() as s:
                    ok, errs = db.review_fix(s, d.id, f, ValidatorAgent())
                st.success("Saved") if ok else st.error("; ".join(errs))
