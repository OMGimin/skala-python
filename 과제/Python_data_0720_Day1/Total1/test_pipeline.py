"""Six unit tests required by the asynchronous ETL capstone guide."""

import asyncio

import pandas as pd

from pipeline import extract, load, transform


def product_row(product_id=1, category=" FOOD ", price=10.0):
    """Create a small raw product dictionary for unit tests."""
    return {
        "id": product_id,
        "name": f"Product {product_id}",
        "category": category,
        "price": price,
    }


def test_category_is_normalized_to_lowercase():
    """Pydantic should trim and lowercase category text."""
    valid, invalid = transform([product_row(category=" FOOD ")])

    assert not invalid
    assert valid[0].category == "food"


def test_negative_price_is_rejected():
    """A product with a negative price should be classified as invalid."""
    valid, invalid = transform([product_row(price=-5)])

    assert len(valid) == 0
    assert len(invalid) == 1
    assert invalid[0]["errors"][0]["loc"] == ("price",)


def test_valid_and_invalid_counts_match_input():
    """Transform must preserve the total number of input records."""
    rows = [
        product_row(product_id=1, price=10),
        product_row(product_id=2, price=-1),
        product_row(product_id=3, category="Electronics", price=30),
    ]

    valid, invalid = transform(rows)

    assert len(valid) + len(invalid) == len(rows)
    assert len(valid) == 2
    assert len(invalid) == 1


def test_extract_respects_concurrency_limit():
    """Semaphore should prevent more than two test requests at once."""
    active = 0
    maximum_active = 0

    async def tracked_fetch(product_id):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return product_row(product_id=product_id)

    collected, failures = asyncio.run(extract(range(1, 7), max_concurrent=2, fetcher=tracked_fetch))

    assert len(collected) == 6
    assert not failures
    assert maximum_active <= 2


def test_extract_retries_a_temporary_failure():
    """A request that succeeds on its third attempt should be recovered."""
    attempts = 0

    async def flaky_fetch(product_id):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary failure")
        return product_row(product_id=product_id)

    collected, failures = asyncio.run(extract([1], max_retries=3, fetcher=flaky_fetch))

    assert attempts == 3
    assert len(collected) == 1
    assert not failures


def test_load_csv_and_parquet_roundtrip(tmp_path):
    """Saved Parquet data should match the DataFrame returned by Load."""
    valid, invalid = transform(
        [
            product_row(product_id=1, category="Food", price=10.5),
            product_row(product_id=2, category="FASHION", price=20.0),
        ]
    )

    dataframe = load(valid, tmp_path)
    parquet_dataframe = pd.read_parquet(tmp_path / "products.parquet")

    assert not invalid
    assert (tmp_path / "products.csv").exists()
    assert (tmp_path / "products.parquet").exists()
    pd.testing.assert_frame_equal(dataframe, parquet_dataframe)
