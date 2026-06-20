# Every Product Search Query Hides 3 Different Questions. I Built a System That Answers All 3.

**How I built a multi-vector semantic search engine that splits user intent before touching the database, using ColBERT, SigLIP, and BGE inside one Qdrant point.**

---

*Tags: AI Engineering | Vector Search | E-Commerce | Machine Learning | RAG*
*Reading time: ~15 minutes*
*Full source code: [github.com/dvy246/qdrant-multivector](https://github.com/dvy246/qdrant-multivector)*

---

Six months ago I was testing a semantic search prototype for a small e-commerce catalog. Standard setup: sentence-transformer, cosine similarity, top-5 results returned per query. Nothing exotic.

I typed: *"Waterproof black hiking boots with good arch support."*

The results came back. Rank 1 was correct: black waterproof hiking boots. Rank 2 was a waterproof jacket. Rank 3 was a pair of black Chelsea boots. Rank 4 was walking shoes described as "supportive."

On paper, every result was semantically close to the query. The model was doing exactly what I'd set it up to do. But if you were a customer who typed that query, you'd bounce. You wanted hiking boots with arch support, not a jacket or Chelsea boots that happen to share a few words with your query.

I kept tweaking things. Tried a better base model. Tried chunking the product descriptions differently. Tried adding metadata to the text before embedding. Results improved slightly, then hit a wall.

The problem wasn't the model. The problem was that I was asking one embedding to simultaneously capture what a product looks like, what its technical specs say, and what customers in reviews think about it. Those are three completely different types of information. Pooling them into one 384-dimensional vector and calling it "the product" was always going to lose signal.

So I rebuilt it. One point in the database. Three separate vector matrices per product. A query decomposer that routes intent before a single embedding gets made. And a personalization layer that reranks results without touching the retrieval stage.

This article walks through the whole thing, including why I picked the tools I did, where it works well, and where it doesn't.

---

## The mental model: before you read a single line of code

Here's the full pipeline in one view. Read this once, keep it in mind, and the rest of the article will click much faster.

```
USER QUERY
"Waterproof black hiking boots with good arch support"
                         │
                 ┌───────▼───────┐
                 │ QUERY DECOMPOSER
                 └───────┬───────┘
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    "waterproof"   "black hiking    "arch support"
                       boots"
          │              │              │
    ┌─────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐
    │  ColBERT   │ │   SigLIP   │ │    BGE    │
    │  96-d tok  │ │  768-d img │ │  384-d    │
    └─────┬──────┘ └─────┬──────┘ └────┬──────┘
          │              │              │
   text_vectors   visual_vectors  review_vectors
          │              │              │
    ┌─────▼──────────────▼──────────────▼─────┐
    │        QDRANT: ONE POINT PER PRODUCT     │
    │  { visual_matrix  [1, 768]               │
    │    text_matrix    [~28, 96]              │
    │    review_matrix  [findings, 384] }      │
    └────────────────┬─────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │ PREFETCH (text+review) │  ← candidate retrieval
         │ FINAL QUERY (visual)   │  ← MAX_SIM scoring
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  PER-ASPECT BREAKDOWN  │
         │  visual = 0.81         │
         │  text   = 0.44         │
         │  review = 0.69         │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  PERSONALIZATION BOOST │  ← brand, price, eco, category
         └───────────┬───────────┘
                     │
              RANKED RESULTS
```

**How to read this:**

The query comes in as one string and splits into three sub-queries before any embedding happens. Each sub-query goes to a different model, which produces a different vector matrix, stored in a different named field on the same Qdrant point. At search time, text and review fields run as prefetch stages to pull candidates. The visual field does the final scoring over those candidates using MAX_SIM. Then a lightweight personalization layer reranks by user preferences. The final result list includes a per-product explanation showing which signal matched.

Every section below is the detail behind one box in this diagram.

---

## The core problem: one vector can't represent everything

When someone searches "waterproof black hiking boots with good arch support," they're not asking one question. They're asking three at the same time:

The first is a spec question: "is this waterproof?" That's answered by product descriptions and technical data sheets.

The second is a visual question: "does it look like black hiking boots?" That's answered by images. Not text descriptions of images. Actual images.

The third is a social question: "do people say the arch support is good?" That's answered by customer reviews.

No single embedding model can capture all three signals with equal fidelity. When you pool them together, you get a vector that's roughly in the neighborhood of all three but isn't precise about any of them. It's why you get Chelsea boots in your hiking boot results. Both are black footwear. The visual signal is doing fine. The arch-support signal is completely lost.

The fix is to stop treating a product as one thing and start treating it as a collection of distinct signals, each searchable independently.

---

## Why I chose Qdrant for this

I evaluated Pinecone, Weaviate, and Qdrant before committing. The deciding factor was named multivector fields.

I needed to store multiple different types of vectors per document, each with its own dimensionality and comparison function. Pinecone doesn't support this natively. Weaviate's multivector story was still evolving when I started. Qdrant's API is clean: you define named vector configs at collection creation time, each with different dimensions, distance metrics, and HNSW settings.

The other thing that sold me was `update_vectors`. In a live catalog:

- Products get new images. The visual matrix needs to grow without a collection rebuild.
- Reviews arrive daily. Each new review adds rows to the review matrix.
- Qdrant handles both by letting you retrieve an existing vector field, concatenate new vectors, and push the updated matrix back. The collection stays up. Running queries keep working.

Versions matter here. I'm on Qdrant `v1.15.3` with `qdrant-client>=1.15.0`. Multivectors have been available since `v1.10`, but the `query_points` prefetch API that makes multi-stage search work cleanly became stable in `v1.14`.

---

## What the system stores: one point, three matrices

Every product becomes a single Qdrant point. That point has three named vector fields:

```python
# src/commerce_engine/qdrant_store.py

VISUAL_VECTOR = "visual_vectors"   # 768-d SigLIP image embedding
TEXT_VECTOR   = "text_vectors"     # 96-d ColBERT token matrix
REVIEW_VECTOR = "review_vectors"   # 384-d BGE per-finding embeddings
```

All three use `MAX_SIM` as the comparator. Here is the full collection setup:

```python
def vector_params(size: int, *, hnsw_m: int | None = None) -> models.VectorParams:
    hnsw_config = None
    if hnsw_m is not None:
        hnsw_config = models.HnswConfigDiff(m=hnsw_m)
    return models.VectorParams(
        size=size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        hnsw_config=hnsw_config,
    )


def recreate_collection(client, collection, *, profile="baseline", disable_text_hnsw=True):
    client.create_collection(
        collection_name=collection,
        vectors_config={
            VISUAL_VECTOR: vector_params(VISION_DIM),                                       # 768
            TEXT_VECTOR:   vector_params(TEXT_DIM, hnsw_m=0 if disable_text_hnsw else None),# 96
            REVIEW_VECTOR: vector_params(REVIEW_DIM),                                       # 384
        },
        quantization_config=quantization_config(profile),
    )
    create_payload_indexes(client, collection)
```

The `hnsw_m=0` on `TEXT_VECTOR` is intentional. Setting `m=0` disables HNSW graph indexing entirely for that field. Normally this would be a performance disaster. Here it is not, because text vectors are used as a ColBERT reranker in the final query stage, not for first-stage ANN retrieval. When ColBERT is scoring over a small candidate set (20-50 products), brute-force matrix comparison is actually faster than HNSW traversal because the graph overhead outweighs the lookup cost at small N. Disabling HNSW also saves index memory, which compounds if you're storing token-level matrices at scale.

The payload also gets indexed separately so Qdrant can filter before scoring:

```python
def create_payload_indexes(client, collection):
    # Keyword fields
    for field in ["brand", "category", "region", "color", "size", "product_id"]:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    # Bool field
    client.create_payload_index(
        collection_name=collection,
        field_name="availability",
        field_schema=models.PayloadSchemaType.BOOL,
    )
    # Float fields
    for field in ["price", "eco_score"]:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.FLOAT,
        )
```

This matters more than it looks. Payload filtering in Qdrant runs before vector scoring when indexes are present. So filtering for `availability=True` or `price <= 150.0` removes candidates before any embedding comparison happens. The alternative is post-filtering, which is slower and produces inconsistent result counts. Always index your filterable fields before you start querying.

---

## The three embedding pipelines

Each vector field in the point comes from a different embedding pipeline. The same product goes through all three before ingestion.

### Visual pipeline: SigLIP

Product images go through `google/siglip-base-patch16-224` from Hugging Face. SigLIP is a vision-language model that produces aligned image and text embeddings. "Aligned" means you can encode a text query like "black hiking boots" and compare it directly to an image embedding without any additional bridge model.

```python
# src/commerce_engine/embeddings.py

def image_patches(self, image_path: Path) -> list[list[float]]:
    """Returns shape [1, 768]: normalized SigLIP pooled image embedding."""
    self._load_siglip()
    import torch

    image = Image.open(image_path).convert("RGB")
    siglip_inputs = self._siglip_processor(images=image, return_tensors="pt")
    siglip_inputs = {k: v.to(self.device) for k, v in siglip_inputs.items()}

    with torch.no_grad():
        vision_outputs = self._siglip_model.vision_model(**siglip_inputs)
        pooled = vision_outputs.pooler_output        # shape: [1, 768]
        normalized = torch.nn.functional.normalize(pooled, dim=-1)

    return normalized.cpu().numpy().astype(float).tolist()
```

And the corresponding text-side query encoding:

```python
def visual_query(self, query: str) -> list[list[float]]:
    """Returns shape [1, 768]: normalized SigLIP text tower output."""
    self._load_siglip()
    import torch

    text_inputs = self._siglip_processor(
        text=[query], return_tensors="pt", padding=True, truncation=True
    )
    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_outputs = self._siglip_model.text_model(**text_inputs)
        pooled = text_outputs.pooler_output          # shape: [1, 768]
        normalized = torch.nn.functional.normalize(pooled, dim=-1)

    return normalized.cpu().numpy().astype(float).tolist()
```

Both outputs are L2-normalized before storage and querying. This makes cosine similarity equivalent to dot product, which is faster for Qdrant to compute.

The stored shape for each product is `[1, 768]`, a matrix with one row. It's technically a 2D multivector with one element, but storing it as a matrix rather than a flat vector keeps the interface consistent and makes appending additional image patches straightforward later.

### Text pipeline: ColBERT

Product titles and spec sheets go through FastEmbed's `answerdotai/answerai-colbert-small-v1`. This model produces token-level embeddings, 96 dimensions per token, without pooling.

```python
def text_late(self, texts: list[str]) -> list[list[list[float]]]:
    return [
        embedding.astype(float).tolist()
        for embedding in self.late_model.embed(texts)
    ]
```

The return shape is `[num_texts, num_tokens, 96]`. For one product title, you get a matrix of shape `[num_tokens, 96]`. A spec-heavy product like "TrailForge StormShield Black Hiking Boots. support: nylon shank and molded arch support. upper: black ripstop textile. waterproof_rating: IPX6 waterproof membrane for heavy rain" tokenizes to around 25-30 tokens, so the stored matrix is `[~28, 96]`.

The text document is assembled in the Product model:

```python
# src/commerce_engine/models.py

def text_document(self) -> str:
    specs = " ".join(f"{key}: {value}" for key, value in sorted(self.specs.items()))
    return f"{self.title}. {specs}"
```

Title first, then all specs as key-value pairs. The spec keys (`support`, `waterproof_rating`, `upper`) are included as text so ColBERT can match a query token like "waterproof" to the spec key, not just the spec value.

### Review pipeline: BGE with semantic finding extraction

Raw reviews go through a finding extraction step before embedding. This is where the architecture differs most from a naive approach.

A customer review like "Waterproof in heavy rain and mud. Good arch support on long hikes. Excellent grip on wet rock." contains three distinct factual claims. If you embed the entire review as a single vector, you get one embedding that's the average of all three signals. If one product has ten reviews and you embed each one whole, you get ten vectors with averaged signals.

Instead, `extract_semantic_findings()` pulls out specific claims using regex patterns:

```python
# src/commerce_engine/reviews.py

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
            findings.append(normalized.rstrip("."))
    return list(dict.fromkeys(findings))  # deduplication
```

The findings for the TrailForge boot come out as: `["waterproof in heavy rain and mud", "good arch support", "excellent grip"]`. Each finding then gets embedded separately with `BAAI/bge-small-en-v1.5` into a 384-dimensional vector:

```python
def review_findings(self, findings: list[str]) -> list[list[float]]:
    if not findings:
        return [_unit_vector("empty-review", REVIEW_DIM)]
    return [
        embedding.astype(float).tolist()
        for embedding in self.review_model.embed(findings)
    ]
```

The stored review matrix is `[num_findings, 384]`. Three findings become a `[3, 384]` matrix. A product with ten reviews and 15 extracted findings becomes `[15, 384]`.

When a query includes "good arch support," it gets embedded as a single 384-d vector and compared via MAX_SIM against this matrix. The most similar finding wins. That matching finding is "good arch support" from a review, not an approximation averaged across all review content.

The full ingestion step assembles all three matrices into a single point:

```python
# src/commerce_engine/ingest.py

def product_point(product: Product, embedder: Embedder) -> models.PointStruct:
    text_matrix   = embedder.text_late([product.text_document()])[0]
    review_matrix = embedder.review_findings(
        extract_semantic_findings(product.reviews)
    )
    visual_matrix = embedder.image_patches(Path(product.image_path))

    return models.PointStruct(
        id=point_id(product.id),
        payload=product.payload(),
        vector={
            VISUAL_VECTOR: visual_matrix,   # shape [1, 768]
            TEXT_VECTOR:   text_matrix,     # shape [~28, 96]
            REVIEW_VECTOR: review_matrix,   # shape [num_findings, 384]
        },
    )
```

One product. One point. Three matrices, each solving a different retrieval problem.

---

## Query decomposition: routing intent before embedding

When a user submits a query, the first thing that happens is decomposition. The query does not go directly to all three vector fields.

```python
# src/commerce_engine/query.py

TEXT_KEYWORDS   = {"waterproof", "water-resistant", "rain", "membrane", "insulated", "leather"}  # trimmed — full set in query.py
VISUAL_KEYWORDS = {"black", "brown", "charcoal", "red", "blue", "hiking", "boots", "sneakers"}  # trimmed — full set in query.py
REVIEW_KEYWORDS = {"support", "arch", "grip", "durable", "comfortable", "runs", "excellent"}    # trimmed — full set in query.py


def decompose_query(query: str) -> QueryPlan:
    tokens = tokenize(query)
    text_terms   = [t for t in tokens if t in TEXT_KEYWORDS]
    visual_terms = [t for t in tokens if t in VISUAL_KEYWORDS]
    review_terms = [t for t in tokens if t in REVIEW_KEYWORDS]

    phrase = " ".join(tokens)
    # phrase-level overrides for known compound terms
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
```

For "Waterproof black hiking boots with good arch support" the decomposition produces:
- `text_terms = ["waterproof"]` → routed to ColBERT spec matching
- `visual_terms = ["black", "hiking", "boots"]` → routed to SigLIP visual matching
- `review_terms = ["support", "arch", "good"]` → routed to BGE review matching

The `QueryPlan` model exposes three properties that join these terms back into sub-query strings:

```python
# src/commerce_engine/models.py

@property
def text_query(self) -> str:
    return " ".join(self.text_terms or [self.original_query])

@property
def visual_query(self) -> str:
    return " ".join(self.visual_terms or [self.original_query])

@property
def review_query(self) -> str:
    return " ".join(self.review_terms or [self.original_query])
```

If no terms match a category, the original query falls back as the sub-query for that field. This prevents empty queries.

I'll be direct about the limitation: this decomposition uses keyword matching, which is coarse. A better production implementation would call an LLM to classify query intent. The architecture is the same either way. What matters is that you route before you embed, not that you route using a fancy classifier. Get the structure right first.

---

## The search: prefetch then final scoring

Qdrant's `query_points` supports a `prefetch` parameter. It lets you retrieve candidate sets from multiple vector fields in parallel, then apply a final scoring stage against only those candidates. This is the mechanism that ties the three vector fields together.

```python
# src/commerce_engine/search.py

def search_products(client, collection, request, embedder):
    plan = decompose_query(request.query)
    query_filter = build_filter(request.filters)

    # Embed all three sub-queries
    text_query   = embedder.text_late([plan.text_query])[0]
    review_query = embedder.review_findings([plan.review_query])
    visual_query = embedder.visual_query(plan.visual_query)

    candidate_limit = max(request.limit * 5, 20)

    # Text and review run as prefetch candidates
    prefetch = [
        models.Prefetch(
            query=text_query,
            using=TEXT_VECTOR,
            limit=candidate_limit,
            filter=query_filter,
        ),
        models.Prefetch(
            query=review_query,
            using=REVIEW_VECTOR,
            limit=candidate_limit,
            filter=query_filter,
        ),
    ]

    # Visual query scores and ranks the prefetch candidates
    response = client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=visual_query,
        using=VISUAL_VECTOR,
        query_filter=query_filter,
        limit=max(request.limit * 3, 10),
        with_payload=True,
        with_vectors=True,
    )
```

Text and review prefetch stages each return up to `candidate_limit` results. Qdrant unions them. The final visual query then scores only those candidates using the `VISUAL_VECTOR` field. Payload filters apply at every stage.

After Qdrant responds, the code computes per-aspect MAX_SIM scores to get a named breakdown:

```python
    from commerce_engine.scoring import maxsim_score

    for point in response.points:
        vectors  = point.vector or {}
        doc_visual = vectors.get(VISUAL_VECTOR, [])
        doc_text   = vectors.get(TEXT_VECTOR, [])
        doc_review = vectors.get(REVIEW_VECTOR, [])

        v_score = maxsim_score(visual_query, doc_visual) if doc_visual else 0.0
        t_score = maxsim_score(text_query,   doc_text)   if doc_text   else 0.0
        r_score = maxsim_score(review_query, doc_review) if doc_review else 0.0

        combined_score = v_score + t_score + r_score
```

The MAX_SIM function itself is three lines of numpy:

```python
# src/commerce_engine/scoring.py

def maxsim_score(query_matrix, doc_matrix) -> float:
    query = np.asarray(query_matrix, dtype=np.float32)
    doc   = np.asarray(doc_matrix,   dtype=np.float32)
    similarities = query @ doc.T                # [query_tokens, doc_tokens]
    return float(similarities.max(axis=1).sum())# max per query token, then sum
```

For each query token, find the highest similarity among all document tokens. Sum those maximums across all query tokens. That sum is the MAX_SIM score. This is the full ColBERT late-interaction formula. Nothing else.

Computing it manually after retrieval serves a purpose: you get per-aspect scores (`visual=0.81, text=0.44, review=0.69`) that go into the explanation field. That breakdown is the single most useful debugging tool in the whole system. When results feel wrong, looking at the per-aspect breakdown tells you immediately whether it's a visual mismatch, a spec mismatch, or a review signal problem.

---

## Personalization: reranking the same candidates differently

After Qdrant returns results, a personalization layer reranks them based on the requesting user's profile. The Qdrant retrieval step is identical for every user. Only the reranking changes.

```python
# src/commerce_engine/scoring.py

def personalization_boost(payload: dict, profile: UserProfile) -> tuple[float, list[str]]:
    boost = 0.0
    reasons = []

    if payload.get("brand") in profile.preferred_brands:
        boost += 0.25
        reasons.append(f"preferred brand: {payload['brand']}")

    price = float(payload.get("price", 0.0))
    low, high = profile.price_range
    if low <= price <= high:
        boost += 0.30
        reasons.append(f"price in user range: {low:g}-{high:g}")

    if profile.eco_preference and payload.get("is_sustainable"):
        boost += 0.30
        reasons.append("eco preference matched")

    if payload.get("category") in profile.favorite_categories:
        boost += 0.20
        reasons.append(f"favorite category: {payload['category']}")

    return boost, reasons


def rerank(results: list[SearchResult], profile: UserProfile) -> list[SearchResult]:
    reranked = []
    for result in results:
        boost, reasons = personalization_boost(result.payload, profile)
        final_score = result.qdrant_score * (1.0 + boost)
        reranked.append(
            result.model_copy(update={
                "final_score": final_score,
                "personalization_boost": boost,
                "explanation": [*result.explanation, *reasons],
            })
        )
    return sorted(reranked, key=lambda r: r.final_score, reverse=True)
```

The boost is multiplicative. A product with a semantic score of 0.90 and a total boost of 0.55 finishes with 0.90 × 1.55 = 1.395. A product with a semantic score of 0.95 and no boosts stays at 0.95. Personalization can flip the ranking.

Every boost reason becomes part of the explanation string. The user sees: `"preferred brand: TrailForge | price in user range: 80-165 | review evidence: arch support"`. That's the kind of explainability that's worth building. Users trust recommendations more when they can see why something was recommended. Debugging becomes faster when you can see exactly which signal contributed to which result.

---

## Updating products without rebuilding anything

Production catalogs are not static. A product gets a new image. Reviews come in daily. The system needs to handle both without downtime.

Both update operations follow the same pattern: retrieve the existing vector field, concatenate new vectors, push the updated matrix back with `update_vectors`.

```python
# src/commerce_engine/updates.py

def append_review(client, collection, product_id, review, embedder):
    point    = _get_point(client, collection, product_id)
    findings = extract_semantic_findings([review])
    new_vecs = embedder.review_findings(findings)

    existing        = (point.vector or {}).get(REVIEW_VECTOR, [])
    updated_matrix  = [*existing, *new_vecs]

    update_named_vectors(client, collection, product_id, {REVIEW_VECTOR: updated_matrix})
    client.set_payload(
        collection_name=collection,
        payload={"reviews": [*point.payload.get("reviews", []), review]},
        points=[point_id(product_id)],
        wait=True,
    )
    return {"product_id": product_id, "findings": findings}


def append_image(client, collection, product_id, image_path, embedder):
    point   = _get_point(client, collection, product_id)
    new_vecs = embedder.image_patches(image_path)

    existing       = (point.vector or {}).get(VISUAL_VECTOR, [])
    updated_matrix = [*existing, *new_vecs]

    update_named_vectors(client, collection, product_id, {VISUAL_VECTOR: updated_matrix})
    return {
        "product_id": product_id,
        "added_patches": len(new_vecs),
        "total_patches": len(updated_matrix),
    }
```

After `append_review`, the product's review matrix grows by one row per extracted finding. After `append_image`, the visual matrix grows by one row. No collection rebuild. Existing search queries keep working during the update.

This is one of the most practical aspects of the design. If adding a review required re-ingesting the whole product, or worse, rebuilding the collection, you'd end up batching updates and your review data would always be stale. With `update_vectors`, you can write a simple webhook that processes reviews in real time.

---

## What the benchmarks actually showed

The benchmark runs 104 queries against the fixture product set (4 hiking boot variants), generated from product attribute combinations: color + category, brand + category, spec-based, review-fragment, and compound queries. Latencies measured with `time.perf_counter` against a live Qdrant instance.

Results for the baseline profile (no quantization, HNSW disabled for text):

| Metric | Value |
|---|---|
| Queries | 104 |
| Mean latency | 2.2 ms |
| P50 latency | 2.2 ms |
| P95 latency | 2.5 ms |
| P99 latency | 2.7 ms |
| Recall@3 | 95.2% |
| Recall@5 | 100% |
| MRR | 0.7204 |
| NDCG@5 | 0.7920 |

95.2% Recall@3 means 99 of 104 queries returned the target product in the top 3. The 5 failures are edge cases where the query decomposer routes everything to one aspect but the product's signal lives in a different field.

Honest caveats worth stating:

- This is 4 products. The 2.2ms will not hold at 100k products.
- Benchmark queries were generated from the same fixture data they retrieve. Some overfitting is baked in.
- Binary quantization knocked Recall@3 down noticeably. Compressing 32-bit floats to 1 bit is extreme for ColBERT's token-level matching. Start with scalar INT8 and measure recall before deploying.

---

## When not to build this

Three embedding pipelines, a query decomposer, and prefetch orchestration is not always the right answer. Skip it if any of these apply:

- **Short, uniform queries** like "red shoes" or "wool sweater." Single dense vector handles this fine.
- **No clean product images.** SigLIP needs consistent product photography. Blurry or inconsistent shots produce weak visual signal. Drop that field.
- **No customer reviews.** The review field adds complexity for zero gain. Skip it.
- **Catalog under 10k products with low query load.** The simpler architecture will likely be fast enough and much cheaper to maintain.
- **Broken baseline retrieval.** Bad base embeddings, wrong chunking, or misconfigured filters won't get fixed by adding more vectors. Multivectors amplify signal. If there's no signal to amplify, they amplify noise.

---

## When this is worth building

The complexity pays off when all of these are true:

- **Queries contain mixed intent.** Spec questions, visual questions, and review-based questions in the same search string. Common in outdoor gear, fashion, electronics, and beauty.
- **Clean product images at scale.** 100k+ SKUs with consistent photography. The SigLIP channel earns its storage cost here.
- **Reviews arrive continuously.** If reviews update daily, the `update_vectors` pattern becomes genuinely valuable versus batch re-indexing.
- **You have user profile data.** The personalization layer needs signal to work with. Without it, the reranking step is adding noise.

---

## Decision matrix

```
╔══════════════════════════════════════════╦════════════════════════════════════════════╗
║ Situation                                ║ What to do                                 ║
╠══════════════════════════════════════════╬════════════════════════════════════════════╣
║ Short, uniform queries                   ║ Single dense vector, done                  ║
║ Long product specs + technical docs      ║ Add ColBERT text multivector, m=0 for HNSW ║
║ Clean product images at scale            ║ Add SigLIP visual multivector              ║
║ Real customer reviews available          ║ Add BGE review multivector per finding     ║
║ All three data types present             ║ Full setup with prefetch staging           ║
║ Need to reduce RAM                       ║ Scalar INT8 first, measure recall          ║
║ Reviews update frequently                ║ Use update_vectors, never rebuild          ║
║ No images or bad image quality           ║ Skip visual field entirely                 ║
║ Sub-10k catalog, low query load          ║ Don't build this yet                       ║
╚══════════════════════════════════════════╩════════════════════════════════════════════╝
```

---

## Implementation path if you're starting from scratch

```
Step 1: Single dense vector for title + specs
        → Verify baseline recall on 50 real queries

Step 2: Add payload indexes and filters
        → Verify filter correctness before adding more vectors

Step 3: Add ColBERT text multivector (96-d)
        → Disable HNSW if using as reranker (m=0)

Step 4: Add review finding extraction
        → Extract claims, not full reviews

Step 5: Add BGE review multivector (384-d)
        → One vector per extracted finding

Step 6: Add SigLIP visual multivector (768-d)
        → Only if clean images exist

Step 7: Add query decomposition
        → Keyword matching first, LLM routing later

Step 8: Add personalization reranker
        → Multiplicative boost over Qdrant scores
```

Do not skip step 1. The single-vector baseline is not just scaffolding. It is the control you need to measure whether the additional complexity actually improves results for your specific catalog. I have seen teams add three embedding pipelines on day one because the architecture sounds impressive, then spend three weeks trying to figure out why recall got worse. It was worse because their base embeddings were bad. Fix the foundation before you add floors.

---

## The thing that actually changed after building this

I went back and ran the original "Waterproof black hiking boots with good arch support" query through the finished system. Rank 1 was the TrailForge StormShield. Rank 2 was the EcoTrek TerraDry. Both actual hiking boots with documented arch support in customer reviews.

The Chelsea boots were gone.

The insight isn't that multivectors are magic. The insight is that the original setup was asking one number to represent three different kinds of evidence, and one number is not enough. The fix was to stop compressing and start separating.

Most search quality problems in production are not model problems. They are signal representation problems. The right model pointed at mixed-up data will keep returning wrong answers. Separating the signals is the work. The models are just the tools.

> "Your search returns technically relevant results that nobody clicks? Ask yourself how many different types of user intent you are compressing into one similarity score."

The full source code, tests, benchmark runner, FastAPI endpoints, Streamlit UI, and real product dataset are all at:

**[github.com/dvy246/qdrant-multivector](https://github.com/dvy246/qdrant-multivector)**

Everything described in this article maps to a real file in `src/commerce_engine`. No placeholder functions, no pseudo-code, no hand-waved implementation details.

---

## Project at a glance

A quick summary of everything this system does, for anyone who wants the short version before going through the code.

**What it is:** A multi-aspect semantic search engine for e-commerce that splits every query into visual, spec, and review signals before retrieval.

**Stack:**
- Vector database: Qdrant `v1.15.3` with `qdrant-client>=1.15.0`
- Visual embeddings: `google/siglip-base-patch16-224` (768-d, via Hugging Face Transformers)
- Text embeddings: `answerdotai/answerai-colbert-small-v1` (96-d, via FastEmbed, ColBERT late interaction)
- Review embeddings: `BAAI/bge-small-en-v1.5` (384-d, via FastEmbed, per-finding)
- API: FastAPI + Typer CLI
- UI: Streamlit
- Python: 3.11+

**What each component does:**

| Component | File | Job |
|---|---|---|
| Collection schema | `qdrant_store.py` | Creates 3 named multivector fields with MAX_SIM |
| Embedding backends | `embeddings.py` | Production (SigLIP + ColBERT + BGE) and deterministic (for tests) |
| Ingestion | `ingest.py` | Converts product → 3 matrices → 1 Qdrant point |
| Query decomposer | `query.py` | Splits query into text/visual/review sub-queries |
| Search | `search.py` | Prefetch (text+review) → final (visual) → per-aspect MAX_SIM |
| Scoring | `scoring.py` | MaxSim formula + personalization boost |
| Updates | `updates.py` | Append review/image vectors without collection rebuild |
| Benchmark | `benchmark.py` | 104-query latency + recall runner across quantization profiles |

**Benchmark results (baseline profile, 4 fixture products):**
- Mean latency: 2.2 ms / P95: 2.5 ms
- Recall@3: 95.2% / Recall@5: 100%
- MRR: 0.7204 / NDCG@5: 0.7920

**Run it yourself:**
```bash
git clone https://github.com/dvy246/qdrant-multivector.git
cd qdrant-multivector
uv sync --extra dev
docker compose up -d qdrant
EMBEDDING_BACKEND=deterministic uv run engine init-qdrant
EMBEDDING_BACKEND=deterministic uv run engine ingest --fixtures
EMBEDDING_BACKEND=deterministic uv run engine search \
  "Waterproof black hiking boots with good arch support" --user user_a
```

---

## References

All facts and version numbers in this article were verified against the primary sources below before writing.

1. **Qdrant vectors and multivectors documentation** — the definitive reference for multivector configuration, `MAX_SIM` comparator behavior, and named vector field setup.
   https://qdrant.tech/documentation/manage-data/vectors/

2. **Qdrant hybrid queries and prefetch API** — documents the `query_points` prefetch parameter used in the multi-stage search flow.
   https://qdrant.tech/documentation/search/hybrid-queries/

3. **Qdrant payload indexing documentation** — covers keyword, bool, and float index schema types used in `create_payload_indexes`.
   https://qdrant.tech/documentation/manage-data/indexing/

4. **Qdrant quantization documentation** — covers scalar INT8 and binary quantization config, `always_ram`, and quantile settings.
   https://qdrant.tech/documentation/manage-data/quantization/

5. **FastEmbed ColBERT documentation** — covers `LateInteractionTextEmbedding`, model selection, and token-matrix output format.
   https://qdrant.tech/documentation/fastembed/fastembed-colbert/

6. **Hugging Face SigLIP model card** — covers `google/siglip-base-patch16-224`, the vision and text tower architecture, and pooler output shape.
   https://huggingface.co/docs/transformers/model_doc/siglip

7. **answerdotai/answerai-colbert-small-v1** — the specific FastEmbed ColBERT model used for 96-dimensional token-level text embeddings.
   https://huggingface.co/answerdotai/answerai-colbert-small-v1

8. **BAAI/bge-small-en-v1.5** — the BGE model used for 384-dimensional review finding embeddings.
   https://huggingface.co/BAAI/bge-small-en-v1.5

9. **Women's E-Commerce Clothing Reviews dataset** — the real product dataset derived from Kaggle, used in the Streamlit demo.
   https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews