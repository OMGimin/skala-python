"""Asynchronous Extract-Transform-Load pipeline for the Day 1 capstone.

The default collector uses deterministic mock data so the assignment works
without internet access. Extract limits concurrency and retries failures,
Transform validates products with Pydantic, and Load writes CSV and Parquet.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from models import Product

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MAX_CONCURRENT = 10
MAX_RETRIES = 3
REQUEST_TIMEOUT = 2.0
BACKOFF_BASE = 0.1

ProductRow = dict[str, Any]
Fetcher = Callable[[int], Awaitable[ProductRow]]


async def mock_fetch(product_id: int) -> ProductRow:
    """Simulate one asynchronous API response without real network access."""
    await asyncio.sleep(0.25)

    categories = (" FOOD ", "Electronics", "FASHION")
    price = -1_000 if product_id % 17 == 0 else product_id * 1_000
    return {
        "id": product_id,
        "name": f" Product {product_id} ",
        "category": categories[product_id % len(categories)],
        "price": price,
    }


async def _fetch_with_retry(
    product_id: int,
    fetcher: Fetcher,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    timeout: float,
) -> ProductRow:
    """Fetch one product with timeout, concurrency control, and backoff."""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            # Release the semaphore before backoff so waiting tasks do not
            # occupy a concurrency slot.
            async with semaphore:
                async with asyncio.timeout(timeout):
                    return await fetcher(product_id)
        except Exception as error:  # isolate transient I/O failures per item
            last_error = error
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(BACKOFF_BASE * (2**attempt))

    raise RuntimeError(f"product {product_id} failed after {max_retries} attempts") from last_error


async def extract(
    ids: Iterable[int],
    max_concurrent: int = MAX_CONCURRENT,
    max_retries: int = MAX_RETRIES,
    timeout: float = REQUEST_TIMEOUT,
    fetcher: Fetcher = mock_fetch,
) -> tuple[list[ProductRow], list[dict[str, Any]]]:
    """Collect products concurrently and isolate final request failures."""
    product_ids = list(ids)
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _fetch_with_retry(
            product_id,
            fetcher,
            semaphore,
            max_retries,
            timeout,
        )
        for product_id in product_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    collected = [result for result in results if not isinstance(result, Exception)]
    failures = [
        {"id": product_id, "error": str(result)}
        for product_id, result in zip(product_ids, results, strict=True)
        if isinstance(result, Exception)
    ]
    return collected, failures


def transform(
    raw: Iterable[ProductRow],
) -> tuple[list[Product], list[dict[str, Any]]]:
    """Validate raw products and split them into valid and invalid lists."""
    valid: list[Product] = []
    invalid: list[dict[str, Any]] = []

    for row in raw:
        try:
            valid.append(Product.model_validate(row))
        except ValidationError as error:
            invalid.append({"data": row, "errors": error.errors()})

    return valid, invalid


def load(valid: Iterable[Product], out_dir: Path | str = OUTPUT_DIR) -> pd.DataFrame:
    """Save valid products to CSV and Parquet and return their DataFrame."""
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = [product.model_dump() for product in valid]
    dataframe = pd.DataFrame(records, columns=["id", "name", "category", "price"])
    dataframe.to_csv(output_path / "products.csv", index=False)
    dataframe.to_parquet(output_path / "products.parquet", index=False)
    return dataframe


async def run(
    ids: Iterable[int],
    out_dir: Path | str = OUTPUT_DIR,
) -> dict[str, int | float]:
    """Orchestrate Extract, Transform, and Load without duplicating their work."""
    product_ids = list(ids)
    started_at = time.perf_counter()

    raw, extract_failures = await extract(product_ids)
    valid, invalid = transform(raw)
    dataframe = load(valid, out_dir)

    return {
        "requested": len(product_ids),
        "extracted": len(raw),
        "extract_failed": len(extract_failures),
        "valid": len(valid),
        "invalid": len(invalid),
        "rows_saved": len(dataframe),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }


if __name__ == "__main__":
    pipeline_summary = asyncio.run(run(range(1, 61)))
    print("Asynchronous ETL completed")
    print(pipeline_summary)
