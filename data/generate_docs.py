"""Seeded synthetic invoices with gold labels. Every 10th document has no usable total (gold = None)."""
import json
import random
from datetime import date, timedelta

VENDORS = ["Acme Supplies Ltd", "Northwind Traders", "Globex Corporation", "Initech Services",
           "Umbrella Logistics", "Stark Industries", "Wayne Enterprises", "Hooli Cloud"]
SYMBOL = {"USD": "$", "EUR": "€", "INR": "₹", "GBP": "£"}
TEMPLATES = [
    "INVOICE\nFrom: {vendor}\nInvoice No: {num}\nDate: {date}\nAmount Due: {total}\nThank you for your business.",
    "{vendor}\nTax Invoice #{num} dated {date}\nItems: consulting, hardware\nTOTAL PAYABLE {total}",
    "Bill from {vendor} | Ref {num} | Issued {date} | Grand total: {total} | Payment within 30 days",
    "Hi team, please find attached invoice {num} from {vendor}, issued on {date}. The total comes to {total}.\nRegards, Accounts",
]


def generate(n=50, seed=11):
    rng, docs = random.Random(seed), []
    for i in range(n):
        vendor, cur = rng.choice(VENDORS), rng.choice(list(SYMBOL))
        d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 330))
        total, num = round(rng.uniform(50, 9000), 2), f"INV-{rng.randint(1000, 9999)}"
        s = f"{total:,.2f}"
        shown_total = [f"{SYMBOL[cur]}{s}", f"{cur} {s}", f"{s} {cur}"][i % 3]
        shown_date = [d.isoformat(), d.strftime("%d %b %Y"), d.strftime("%B %d, %Y")][i % 3]
        incomplete = i % 10 == 9
        text = TEMPLATES[i % 4].format(vendor=vendor, num=num, date=shown_date,
                                       total="to be confirmed" if incomplete else shown_total)
        gold = None if incomplete else {"vendor": vendor, "invoice_number": num,
                                        "invoice_date": d.isoformat(), "total": total, "currency": cur}
        docs.append({"id": i, "text": text, "gold": gold})
    return docs


if __name__ == "__main__":
    docs = generate()
    json.dump(docs, open("eval/labeled_docs.json", "w"), indent=1, ensure_ascii=False)
    print(len(docs), "documents written")
