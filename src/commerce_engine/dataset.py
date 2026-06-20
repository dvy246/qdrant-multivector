"""Real-world product dataset loader.

Loads a curated 20-product subset derived from the Women's E-Commerce
Clothing Reviews dataset (Kaggle) stored as ``data/real_products.json``.
Fixture data in ``fixtures.py`` remains unchanged and is used exclusively
by the test suite and benchmark runner.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from commerce_engine.models import Product, UserProfile

# ---------------------------------------------------------------------------
# Demo user profiles aligned with real-dataset brands and categories
# ---------------------------------------------------------------------------

DEMO_USERS: dict[str, UserProfile] = {
    "fashion_forward": UserProfile(
        id="fashion_forward",
        preferred_brands=["Reformation", "Anthropologie", "Free People"],
        price_range=(60.0, 250.0),
        eco_preference=True,
        favorite_categories=["dresses", "tops"],
    ),
    "casual_classic": UserProfile(
        id="casual_classic",
        preferred_brands=["Madewell", "J.Crew", "Everlane"],
        price_range=(30.0, 150.0),
        eco_preference=False,
        favorite_categories=["jeans", "tops", "pants"],
    ),
    "active_minimalist": UserProfile(
        id="active_minimalist",
        preferred_brands=["Lululemon", "Patagonia", "Everlane"],
        price_range=(50.0, 200.0),
        eco_preference=True,
        favorite_categories=["activewear", "jackets"],
    ),
}

# ---------------------------------------------------------------------------
# Color palette for product image generation
# ---------------------------------------------------------------------------

_IMAGE_COLORS: dict[str, tuple[int, int, int]] = {
    "burgundy": (128, 0, 32),
    "black": (20, 20, 25),
    "indigo": (44, 62, 100),
    "sage": (138, 154, 91),
    "cream": (245, 240, 220),
    "white": (245, 245, 248),
    "olive": (107, 142, 35),
    "charcoal": (54, 69, 79),
    "navy": (28, 42, 82),
    "blush": (222, 166, 167),
    "navy stripe": (28, 42, 82),
    "beige": (210, 180, 140),
    "rust": (183, 65, 14),
    "floral": (219, 112, 147),
    "dark olive": (85, 107, 47),
}


def _product_color(color_name: str) -> tuple[int, int, int]:
    """Return an RGB tuple for the product colour, with a fallback."""
    return _IMAGE_COLORS.get(color_name.lower(), (120, 120, 130))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_raw_products(root: Path) -> list[dict]:
    """Read the JSON product catalog from disk."""
    json_path = root / "data" / "real_products.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Real product dataset not found at {json_path}. "
            "Ensure data/real_products.json exists in the project root."
        )
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def load_real_products(root: Path) -> list[Product]:
    """Load the real product dataset and generate placeholder images.

    Returns a list of ``Product`` instances with ``image_path`` populated.
    """
    raw = _load_raw_products(root)
    image_dir = root / "data" / "images"
    try:
        image_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        image_dir = Path("/Users/divyyadav/.gemini/antigravity-ide/scratch/images")
        image_dir.mkdir(parents=True, exist_ok=True)

    products: list[Product] = []
    for item in raw:
        product = Product(**item)
        path = image_dir / f"{product.id}.png"
        if not path.exists():
            try:
                _generate_product_image(product, path)
            except PermissionError:
                alt_dir = Path("/Users/divyyadav/.gemini/antigravity-ide/scratch/images")
                alt_dir.mkdir(parents=True, exist_ok=True)
                path = alt_dir / f"{product.id}.png"
                if not path.exists():
                    _generate_product_image(product, path)
        products.append(product.model_copy(update={"image_path": path}))
    return products



def _generate_product_image(product: Product, path: Path) -> None:
    """Create a simple branded placeholder image for a product."""
    bg = (240, 240, 235)
    fg = _product_color(product.color)

    image = Image.new("RGB", (224, 224), bg)
    draw = ImageDraw.Draw(image)

    # Draw a simple garment silhouette
    draw.rounded_rectangle((34, 54, 190, 150), radius=18, fill=fg)
    draw.rounded_rectangle((50, 30, 174, 70), radius=12, fill=fg)

    # Brand and category text
    draw.text((34, 164), product.brand, fill=(10, 10, 10))
    draw.text((34, 184), product.category.upper(), fill=(80, 80, 80))

    # Subtle accent stripe at bottom
    accent = tuple(min(c + 40, 255) for c in fg)
    draw.rectangle((0, 216, 224, 224), fill=accent)

    image.save(path)
