from __future__ import annotations

import re

FINDING_PATTERNS = [
    r"waterproof(?: in [a-z ]+)?",
    r"kept feet dry",
    r"good arch support",
    r"arch support is [a-z]+",
    r"supportive footbed",
    r"excellent grip",
    r"not enough grip",
    r"highly durable",
    r"runs a little small",
    r"runs small",
    r"strong ankle support",
]


def extract_semantic_findings(reviews: list[str]) -> list[str]:
    findings: list[str] = []
    for review in reviews:
        normalized = review.lower().strip()
        matched = False
        for pattern in FINDING_PATTERNS:
            match = re.search(pattern, normalized)
            if match:
                findings.append(match.group(0))
                matched = True
        if not matched:
            sentence = normalized.rstrip(".")
            if sentence:
                findings.append(sentence)
    return list(dict.fromkeys(findings))
