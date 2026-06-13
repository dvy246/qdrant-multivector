# AUDIT REPORT — Multi-Aspect E-Commerce Semantic Engine

**Date:** 2026-06-13
**Auditor:** Senior Staff AI Engineer / Principal Architect
**Repository:** `qdrant-multivector-commerce`

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│  Streamlit UI  ←→  FastAPI REST API  ←→  Typer CLI                  │
└──────────────────┬───────────────────────────────┬───────────────────┘
                   │                               │
┌──────────────────▼───────────────────────────────▼───────────────────┐
│                       SEARCH PIPELINE                                │
│  Query → decompose_query() → embed_aspects() → query_points()       │
│       → build_filter() → rerank() → return SearchResult[]           │
└──────────────────┬───────────────────────────────┬───────────────────┘
                   │                               │
┌──────────────────▼───────────┐   ┌───────────────▼───────────────────┐
│     EMBEDDING LAYER          │   │         QDRANT LAYER              │
│  ColBERT (text, 96d)         │   │  Collection: 3 named multivectors │
│  BGE-small (review, 384d)    │   │  visual_vectors: 768d MAX_SIM     │
│  ViT (vision, 768d)          │   │  text_vectors:   96d  MAX_SIM     │
│  DeterministicEmbedder       │   │  review_vectors: 384d MAX_SIM     │
└──────────────────────────────┘   │  Payload indexes: 9 fields        │
                                   └───────────────────────────────────┘
```

**Files:** 17 source files, 2 test files, 1 Streamlit UI, config + Docker.

---

## 2. Data Flow

```
Product(title, specs, reviews, image)
    │
    ├─ text_document() → ColBERT → token matrix [N×96]  → text_vectors
    ├─ extract_findings(reviews) → BGE → finding matrix [M×384] → review_vectors
    └─ image_path → ViT/SigLIP → patch matrix [P×768]  → visual_vectors
    │
    └─ payload dict (brand, price, availability, region, color, sizes, eco_score, ...)
    │
    └─ PointStruct(id=uuid5, vector={3 named matrices}, payload={...})
         │
         └─ client.upsert() → Qdrant
```

---

## 3. Search Flow

```
User Query: "Waterproof black hiking boots with good arch support"
    │
    ├─ decompose_query() → QueryPlan(text=["waterproof"], visual=["black","hiking","boots"], review=["good","arch","support"])
    │
    ├─ embed text_query  → ColBERT → [T×96]
    ├─ embed review_query → BGE   → [1×384]
    ├─ embed visual_query → ViT/hash → [V×768]    ← ⚠️ BROKEN IN PRODUCTION
    │
    ├─ Prefetch(text_vectors, filter)   ──┐
    ├─ Prefetch(review_vectors, filter) ──┼─→ query_points(visual_vectors, final stage)
    │                                      │
    ├─ build_filter(SearchFilters)         │
    │                                      │
    └─ rerank(results, UserProfile) → sorted by final_score = qdrant_score × (1 + boost)
```

---

## 4. Embedding Flow

| Aspect | Model | Dimensions | Backend | Status |
|--------|-------|-----------|---------|--------|
| Text/Specs | ColBERT (`answerai-colbert-small-v1`) | 96 | FastEmbed | ✅ Working |
| Reviews | BGE-small (`bge-small-en-v1.5`) | 384 | FastEmbed | ✅ Working |
| Vision (doc) | ViT (`vit-base-patch16-224-in21k`) | 768 | Transformers | ✅ Working |
| Vision (query) | Hash-based `_unit_vector()` | 768 | Manual | ❌ **NOT ALIGNED** |

---

## 5. Ingestion Flow

```
ensure_fixture_images() → 4 products with 224×224 PNG images
    → product_point(product, embedder) per product
        → text_late(), review_findings(), image_patches()
        → PointStruct with 3 named vector matrices
    → upsert_products() → client.upsert(wait=True)
```

**Issue:** CLI only supports `--fixtures` mode. API accepts arbitrary `list[Product]`.

---

## 6. Update Flow

```
append_review(product_id, review_text):
    → client.retrieve(with_vectors=True)
    → extract_semantic_findings([review])
    → embedder.review_findings(findings)
    → [...existing_review_vectors, ...new_vectors]
    → client.update_vectors()
    → client.set_payload(reviews=[...existing, review])

append_image(product_id, image_path):
    → client.retrieve(with_vectors=True)
    → embedder.image_patches(image_path)
    → [...existing_visual_vectors, ...new_vectors]
    → client.update_vectors()
```

**Correctness:** ✅ Uses `update_vectors`, not `upsert`. No rebuild.

---

## 7. Benchmark Flow

```
run_benchmark(profile):
    → recreate_collection(benchmark_collection, profile)
    → ingest 4 fixture products
    → run 3 queries with expected answers
    → measure latency per query
    → compute recall@3
    → estimate storage bytes analytically
```

---

## 8. IDENTIFIED ISSUES

### Critical

| # | Issue | File | Line | Impact |
|---|-------|------|------|--------|
| C1 | **Visual query embeddings are synthetic** — production `visual_query()` uses `_unit_vector()` hashes, not a multimodal encoder. Text queries cannot semantically match ViT image patches. | `embeddings.py` | 130–136 | Visual retrieval is non-functional in production |
| C2 | **Query decomposition is hardcoded keyword lists** — 28 keywords, no semantic understanding. Any query with unknown words falls through to default. | `query.py` | 7–38 | Brittle, non-generalizable routing |
| C3 | **Review extraction is regex-only** — 11 patterns. Reviews with novel phrasing are captured as raw sentences, losing semantic structure. | `reviews.py` | 5–17 | Limited review intelligence |

### High

| # | Issue | File | Line | Impact |
|---|-------|------|------|--------|
| H1 | **Benchmark uses only 3 queries** — statistically meaningless latency/recall. | `benchmark.py` | 44–48 | Benchmark results unreliable |
| H2 | **Storage measurement is estimated**, not from Qdrant API. | `benchmark.py` | 82–91 | Inaccurate storage reporting |
| H3 | **Only 7 tests, no integration or e2e tests.** No Qdrant-connected test. No API endpoint tests beyond `/health`. | `tests/` | — | Low coverage |
| H4 | **No structured logging.** No request IDs. No latency tracking. | All files | — | No production observability |
| H5 | **Personalization limited to 4 boost factors** (brand, price, eco, category). No behavior history. | `scoring.py` | 19–41 | Shallow personalization |
| H6 | **Streamlit is in main dependencies** instead of optional-only. | `pyproject.toml` | 20 | Unnecessarily heavy install |

### Medium

| # | Issue | File | Line | Impact |
|---|-------|------|------|--------|
| M1 | **No /ready or /metrics endpoints.** Health check doesn't verify Qdrant connectivity. | `api.py` | 44–46 | Deployment blind spot |
| M2 | **No startup validation.** App doesn't fail-fast if Qdrant is unreachable or models missing. | `config.py` | — | Silent failures |
| M3 | **Single-stage Dockerfile.** Contains build tools in final image. | `Dockerfile` | — | Larger image |
| M4 | **No Makefile.** | — | — | Developer friction |
| M5 | **Search explanations are partially hardcoded** route strings, not dynamic per-aspect match reasons. | `search.py` | 70–73 | Explanations not automatically generated |
| M6 | **`maxsim_score()` is a reference utility but never called in the actual search pipeline.** Qdrant handles MaxSim natively. | `scoring.py` | 8–16 | Dead code (acceptable as reference) |
| M7 | **No API error response models.** No OpenAPI descriptions. | `api.py` | — | Incomplete API documentation |

### Low

| # | Issue | File | Line | Impact |
|---|-------|------|------|--------|
| L1 | **`dependencies()` in api.py creates new client/embedder on every request.** Should be app-level singletons. | `api.py` | 31–41 | Wasted connections |
| L2 | **No rate limiting support hooks.** | `api.py` | — | Production risk at scale |
| L3 | **Scalability documentation missing.** | — | — | Unknown scaling profile |

---

## 9. Improvement Plan

| Phase | Description | Files Affected | Priority |
|-------|-------------|----------------|----------|
| 2 | Replace visual query with SigLIP multimodal encoder | `embeddings.py`, `config.py` | **Critical** |
| 3 | Embedding-based query decomposition replacing keyword lists | `query.py`, `models.py` | **Critical** |
| 4 | Semantic review extraction using embeddings | `reviews.py` | **High** |
| 5 | Extended personalization with behavior profiles | `models.py`, `scoring.py`, `fixtures.py` | **High** |
| 6 | Statistical benchmark framework (100+ queries, NDCG, MRR) | `benchmark.py` | **High** |
| 7 | Full test suite (unit/integration/e2e, ≥90% coverage) | `tests/` | **High** |
| 8 | Structured JSON logging with request/correlation IDs | New: `logging_config.py`, all modules | **Medium** |
| 9 | Production configuration with startup validation | `config.py`, `.env.example` | **Medium** |
| 10 | Multi-stage Docker, Makefile, production compose | `Dockerfile`, `docker-compose.yml`, `Makefile` | **Medium** |
| 11 | API improvements: /ready, /metrics, response models, OpenAPI | `api.py` | **Medium** |
| 12 | Automatic search explanations per aspect | `search.py`, `scoring.py` | **Medium** |
| 13 | Scalability guide | New: `SCALING_GUIDE.md` | **Low** |
| 14 | README rewrite to publication quality | `README.md` | **Low** |
