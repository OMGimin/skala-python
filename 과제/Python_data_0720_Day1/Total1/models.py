"""Pydantic v2 models for the asynchronous ETL pipeline."""

from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    """Validate and normalize one product collected by the pipeline."""

    id: int = Field(gt=0)
    name: str = Field(min_length=1)
    category: str
    price: float = Field(gt=0)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Remove surrounding whitespace and reject an empty product name."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must not be empty")
        return normalized

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        """Normalize category text to a non-empty lowercase value."""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("category must not be empty")
        return normalized
