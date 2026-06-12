from __future__ import annotations

import re

from commerce_engine.models import QueryPlan

TEXT_KEYWORDS = {
    "waterproof",
    "water-resistant",
    "rain",
    "membrane",
    "insulated",
    "leather",
    "spec",
    "rating",
}
VISUAL_KEYWORDS = {
    "black",
    "brown",
    "charcoal",
    "red",
    "blue",
    "hiking",
    "boots",
    "sneakers",
    "jacket",
    "pattern",
}
REVIEW_KEYWORDS = {
    "support",
    "arch",
    "grip",
    "durable",
    "runs",
    "small",
    "comfortable",
    "good",
    "excellent",
}


def tokenize(query: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", query.lower())


def decompose_query(query: str) -> QueryPlan:
    tokens = tokenize(query)
    text_terms = [token for token in tokens if token in TEXT_KEYWORDS]
    visual_terms = [token for token in tokens if token in VISUAL_KEYWORDS]
    review_terms = [token for token in tokens if token in REVIEW_KEYWORDS]

    phrase = " ".join(tokens)
    if "good arch support" in phrase:
        review_terms.extend(["good", "arch", "support"])
    if "black hiking boots" in phrase:
        visual_terms.extend(["black", "hiking", "boots"])
    if "waterproof" in phrase:
        text_terms.append("waterproof")

    return QueryPlan(
        original_query=query,
        text_terms=list(dict.fromkeys(text_terms)),
        visual_terms=list(dict.fromkeys(visual_terms)),
        review_terms=list(dict.fromkeys(review_terms)),
    )
