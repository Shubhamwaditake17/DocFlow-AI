from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class InvoiceFields(BaseModel):
    vendor: str = Field(min_length=2)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    total: float = Field(gt=0)
    currency: Literal["USD", "EUR", "INR", "GBP"]

    @field_validator("vendor", "invoice_number")
    @classmethod
    def strip(cls, v):
        return v.strip()


FIELDS = list(InvoiceFields.model_fields)
