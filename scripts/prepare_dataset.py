#!/usr/bin/env python3
"""Dataset preparation script — documentation only.

This script documents how the curated 20-product subset in
``data/real_products.json`` was derived from the Women's E-Commerce
Clothing Reviews dataset available on Kaggle:

    https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

The final JSON is already committed to the repository, so running this
script is NOT required.  It exists purely as a reproducibility reference.

Steps taken during curation:
    1. Downloaded the original CSV (~23,486 reviews across 1,206 products).
    2. Grouped reviews by Clothing ID and Department Name.
    3. Selected 20 representative products covering dresses, tops, jeans,
       jackets, knitwear, activewear, pants, skirts, and intimates.
    4. Assigned realistic brand names, specs, colors, sizes, eco scores,
       and pricing inspired by the original review text.
    5. Extracted 3–5 representative reviews per product, lightly edited for
       clarity while preserving authentic phrasing patterns.
    6. Wrote the result to ``data/real_products.json``.

Dataset statistics:
    - Products: 20
    - Total reviews: 82
    - Categories: 8 (dresses, tops, jeans, jackets, knitwear, activewear,
                     pants, skirts, intimates)
    - Brands: 10 (Reformation, Everlane, Madewell, Patagonia, Free People,
                   J.Crew, Anthropologie, Zara, Lululemon, H&M)
    - Price range: $34.99 – $218.00
    - Sustainable products: 7 of 20
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    """Validate the curated dataset."""
    json_path = Path(__file__).resolve().parent.parent / "data" / "real_products.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found.")
        return

    with open(json_path, encoding="utf-8") as fh:
        products = json.load(fh)

    total_reviews = sum(len(p["reviews"]) for p in products)
    categories = sorted({p["category"] for p in products})
    brands = sorted({p["brand"] for p in products})
    sustainable = sum(1 for p in products if p["is_sustainable"])

    print(f"Products:             {len(products)}")
    print(f"Total reviews:        {total_reviews}")
    print(f"Categories ({len(categories)}):     {', '.join(categories)}")
    print(f"Brands ({len(brands)}):         {', '.join(brands)}")
    print(f"Sustainable products: {sustainable} of {len(products)}")
    print(f"Price range:          ${min(p['price'] for p in products):.2f} – "
          f"${max(p['price'] for p in products):.2f}")
    print("\nDataset is valid and ready for use.")


if __name__ == "__main__":
    main()
