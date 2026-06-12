from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from commerce_engine.models import Product, UserProfile

FIXTURE_PRODUCTS = [
    Product(
        id="boot-001",
        title="TrailForge StormShield Black Hiking Boots",
        brand="TrailForge",
        category="hiking boots",
        price=149.0,
        availability=True,
        region="US",
        color="black",
        sizes=["8", "9", "10", "11"],
        eco_score=0.62,
        is_sustainable=False,
        specs={
            "waterproof_rating": "IPX6 waterproof membrane for heavy rain",
            "upper": "black ripstop textile with reinforced toe",
            "support": "nylon shank and molded arch support",
        },
        reviews=[
            "Waterproof in heavy rain and mud.",
            "Good arch support on long hikes.",
            "Excellent grip on wet rock.",
        ],
    ),
    Product(
        id="boot-002",
        title="EcoTrek TerraDry Charcoal Hiking Boots",
        brand="EcoTrek",
        category="hiking boots",
        price=179.0,
        availability=True,
        region="US",
        color="charcoal",
        sizes=["7", "8", "9", "10"],
        eco_score=0.94,
        is_sustainable=True,
        specs={
            "waterproof_rating": "sealed waterproof bootie",
            "upper": "recycled charcoal textile and plant-based coating",
            "support": "comfort footbed with medium arch support",
        },
        reviews=[
            "Supportive footbed and durable seams.",
            "Runs a little small.",
            "Kept feet dry during a rainy weekend.",
        ],
    ),
    Product(
        id="shoe-003",
        title="CityStride Black Travel Sneakers",
        brand="UrbanWay",
        category="sneakers",
        price=89.0,
        availability=True,
        region="US",
        color="black",
        sizes=["8", "9", "10", "11", "12"],
        eco_score=0.45,
        is_sustainable=False,
        specs={
            "waterproof_rating": "water resistant coating for light rain",
            "upper": "black knit",
            "support": "soft foam insole",
        },
        reviews=[
            "Comfortable for airports.",
            "Not enough grip for trails.",
            "Arch support is mild.",
        ],
    ),
    Product(
        id="boot-004",
        title="AlpinePro Granite Brown Mountaineering Boots",
        brand="AlpinePro",
        category="mountaineering boots",
        price=229.0,
        availability=False,
        region="EU",
        color="brown",
        sizes=["9", "10", "11"],
        eco_score=0.51,
        is_sustainable=False,
        specs={
            "waterproof_rating": "full grain leather with waterproof lining",
            "upper": "brown leather",
            "support": "stiff shank for crampon compatibility",
        },
        reviews=[
            "Highly durable in snow.",
            "Too stiff for casual hiking.",
            "Strong ankle support.",
        ],
    ),
]

FIXTURE_USERS = {
    "user_a": UserProfile(
        id="user_a",
        preferred_brands=["TrailForge", "AlpinePro"],
        price_range=(80.0, 165.0),
        eco_preference=False,
        favorite_categories=["hiking boots"],
    ),
    "user_b": UserProfile(
        id="user_b",
        preferred_brands=["EcoTrek"],
        price_range=(100.0, 220.0),
        eco_preference=True,
        favorite_categories=["hiking boots", "trail gear"],
    ),
}


def ensure_fixture_images(root: Path) -> list[Product]:
    image_dir = root / "data" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "boot-001": (20, 22, 24),
        "boot-002": (58, 62, 60),
        "shoe-003": (16, 17, 19),
        "boot-004": (95, 66, 42),
    }
    products: list[Product] = []
    for product in FIXTURE_PRODUCTS:
        path = image_dir / f"{product.id}.png"
        if not path.exists():
            image = Image.new("RGB", (224, 224), (240, 240, 235))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((34, 84, 190, 142), radius=24, fill=colors[product.id])
            draw.rectangle((66, 66, 132, 90), fill=colors[product.id])
            draw.line((42, 148, 182, 148), fill=(30, 30, 30), width=8)
            draw.text((34, 168), product.brand, fill=(10, 10, 10))
            image.save(path)
        products.append(product.model_copy(update={"image_path": path}))
    return products
