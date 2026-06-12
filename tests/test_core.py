from pathlib import Path

import pytest

from commerce_engine.embeddings import REVIEW_DIM, TEXT_DIM, VISION_DIM, DeterministicEmbedder
from commerce_engine.filters import build_filter
from commerce_engine.fixtures import FIXTURE_USERS, ensure_fixture_images
from commerce_engine.ids import point_id
from commerce_engine.ingest import product_point
from commerce_engine.models import SearchFilters, SearchResult
from commerce_engine.query import decompose_query
from commerce_engine.reviews import extract_semantic_findings
from commerce_engine.scoring import maxsim_score, personalization_boost, rerank


def test_query_decomposition_required_example():
    plan = decompose_query("Waterproof black hiking boots with good arch support")
    assert plan.text_terms == ["waterproof"]
    assert plan.visual_terms == ["black", "hiking", "boots"]
    assert plan.review_terms == ["good", "arch", "support"]


def test_review_findings_extraction():
    findings = extract_semantic_findings([
        "Waterproof in heavy rain and mud.",
        "Good arch support on long hikes.",
        "Excellent grip on wet rock.",
    ])
    assert "waterproof in heavy rain and mud" in findings
    assert "good arch support" in findings
    assert "excellent grip" in findings


def test_filter_builds_compound_conditions():
    qfilter = build_filter(
        SearchFilters(
            min_price=100,
            max_price=200,
            availability=True,
            brands=["TrailForge"],
            colors=["black"],
            sizes=["10"],
        )
    )
    assert qfilter is not None
    assert len(qfilter.must) == 5


def test_maxsim_reference_score():
    score = maxsim_score([[1.0, 0.0], [0.0, 1.0]], [[0.9, 0.1], [0.2, 0.8]])
    assert score == pytest.approx(1.7)


def test_personalization_changes_ranking():
    base_payload_a = {
        "product_id": "boot-001",
        "title": "TrailForge boot",
        "brand": "TrailForge",
        "category": "hiking boots",
        "price": 149,
        "is_sustainable": False,
    }
    base_payload_b = {
        "product_id": "boot-002",
        "title": "EcoTrek boot",
        "brand": "EcoTrek",
        "category": "hiking boots",
        "price": 179,
        "is_sustainable": True,
    }
    results = [
        SearchResult(
            product_id="boot-001",
            title="TrailForge boot",
            brand="TrailForge",
            category="hiking boots",
            qdrant_score=1.0,
            final_score=1.0,
            personalization_boost=0,
            explanation=[],
            payload=base_payload_a,
        ),
        SearchResult(
            product_id="boot-002",
            title="EcoTrek boot",
            brand="EcoTrek",
            category="hiking boots",
            qdrant_score=0.99,
            final_score=0.99,
            personalization_boost=0,
            explanation=[],
            payload=base_payload_b,
        ),
    ]
    assert rerank(results, FIXTURE_USERS["user_a"])[0].product_id == "boot-001"
    assert rerank(results, FIXTURE_USERS["user_b"])[0].product_id == "boot-002"
    boost, reasons = personalization_boost(base_payload_b, FIXTURE_USERS["user_b"])
    assert boost > 0
    assert "eco preference matched" in reasons


def test_product_point_has_named_multivectors(tmp_path: Path):
    product = ensure_fixture_images(tmp_path)[0]
    point = product_point(product, DeterministicEmbedder())
    assert point.id == point_id(product.id)
    assert set(point.vector) == {"visual_vectors", "text_vectors", "review_vectors"}
    assert len(point.vector["visual_vectors"][0]) == VISION_DIM
    assert len(point.vector["text_vectors"][0]) == TEXT_DIM
    assert len(point.vector["review_vectors"][0]) == REVIEW_DIM
