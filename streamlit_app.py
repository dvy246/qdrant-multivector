"""
Multi-Aspect E-Commerce Semantic Engine — Streamlit UI
=====================================================
Visual interface for searching, managing, and benchmarking
the Qdrant multivector commerce search engine.

Run:
    PYTHONPATH=src streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the src package is importable regardless of editable-install state
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from commerce_engine.benchmark import result_dict, run_benchmark
from commerce_engine.dataset import DEMO_USERS, load_real_products
from commerce_engine.embeddings import create_embedder
from commerce_engine.fixtures import FIXTURE_USERS
from commerce_engine.ingest import ingest_products
from commerce_engine.models import SearchFilters, SearchRequest
from commerce_engine.qdrant_store import make_client, recreate_collection
from commerce_engine.query import decompose_query
from commerce_engine.search import search_products
from commerce_engine.updates import append_image, append_review

# Register demo users into the shared user registry so that search.py's
# _profile() function can resolve them without modifying search.py.
FIXTURE_USERS.update(DEMO_USERS)


st.set_page_config(
    page_title="Multi-Aspect Commerce Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --accent: #6C63FF;
    --accent-light: #9D8FFF;
    --teal: #00D4AA;
    --teal-light: #00F5CC;
    --coral: #FF6B6B;
    --orange: #FFA500;
}

.stApp { font-family: 'Inter', sans-serif; }

/* Hero header with gradient text */
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent) 0%, var(--teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem;
    line-height: 1.3;
}
.hero-sub { font-size: 0.95rem; opacity: 0.6; margin-bottom: 1.2rem; }

/* Route badges (query decomposition) */
.route-badge {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.15rem 0.15rem 0.15rem 0;
}
.route-text   { background: rgba(108,99,255,0.12); color: var(--accent-light);
                border: 1px solid rgba(108,99,255,0.25); }
.route-visual { background: rgba(255,165,0,0.12);  color: var(--orange);
                border: 1px solid rgba(255,165,0,0.25); }
.route-review { background: rgba(0,212,170,0.12);  color: var(--teal);
                border: 1px solid rgba(0,212,170,0.25); }

/* Small tag badges */
.tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-right: 0.25rem;
}
.tag-brand    { color: var(--accent-light); border: 1px solid rgba(108,99,255,0.3);
                background: rgba(108,99,255,0.1); }
.tag-cat      { color: var(--orange); border: 1px solid rgba(255,165,0,0.2);
                background: rgba(255,165,0,0.08); }
.tag-ok       { color: var(--teal); border: 1px solid rgba(0,212,170,0.3);
                background: rgba(0,212,170,0.1); }
.tag-out      { color: var(--coral); border: 1px solid rgba(255,107,107,0.3);
                background: rgba(255,107,107,0.1); }
.tag-eco      { color: var(--teal); border: 1px solid rgba(0,212,170,0.3);
                background: rgba(0,212,170,0.1); }

/* Score bars */
.sbar-wrap { margin: 0.35rem 0; }
.sbar-lbl  { font-size: 0.72rem; opacity: 0.65; margin-bottom: 2px; }
.sbar-track{ height: 5px; border-radius: 3px; background: rgba(128,128,128,0.15);
             overflow: hidden; }
.sbar-fill { height: 100%; border-radius: 3px; }
.sf-qdrant { background: linear-gradient(90deg, var(--accent), var(--accent-light)); }
.sf-final  { background: linear-gradient(90deg, var(--teal), var(--teal-light)); }
.sf-boost  { background: linear-gradient(90deg, var(--coral), #FF9F9F); }

/* Price highlight */
.price { font-size: 1.25rem; font-weight: 700; color: var(--teal); }

/* Status dot */
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
.dot-on  { background:var(--teal);  box-shadow:0 0 6px rgba(0,212,170,0.5); }
.dot-off { background:var(--coral); box-shadow:0 0 6px rgba(255,107,107,0.5); }

/* Section dividers */
.sec-hdr {
    font-size: 1rem; font-weight: 600; opacity: 0.85;
    border-bottom: 2px solid rgba(128,128,128,0.15);
    padding-bottom: 0.4rem; margin: 1.4rem 0 0.8rem 0;
}

/* Hide default branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def _make_client(url: str):
    from qdrant_client import QdrantClient
    try:
        c = make_client(url)
        c.get_collections()
        c.is_memory_fallback = False
        return c
    except Exception:
        c = QdrantClient(location=":memory:")
        c.is_memory_fallback = True
        # Initialize collection and pre-populate with real products
        from commerce_engine.qdrant_store import recreate_collection
        from commerce_engine.ingest import ingest_products
        from commerce_engine.embeddings import create_embedder
        
        recreate_collection(c, "commerce_products")
        products = load_real_products(Path.cwd())
        embedder = create_embedder(
            "deterministic",
            text_model="answerdotai/answerai-colbert-small-v1",
            review_model="BAAI/bge-small-en-v1.5",
        )
        ingest_products(c, "commerce_products", products, embedder)
        return c



@st.cache_resource
def _make_embedder(backend: str):
    return create_embedder(
        backend,
        text_model="answerdotai/answerai-colbert-small-v1",
        review_model="BAAI/bge-small-en-v1.5",
    )


@st.cache_resource
def _load_products():
    return load_real_products(Path.cwd())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qdrant_ok(client) -> bool:
    try:
        client.get_collections()
        return True
    except Exception:
        return False


def _collection_ok(client, col: str) -> bool:
    try:
        return client.collection_exists(collection_name=col)
    except Exception:
        return False


def _point_count(client, col: str) -> int:
    try:
        return client.get_collection(collection_name=col).points_count
    except Exception:
        return 0


def _score_bar(label: str, value: float, cap: float, css: str) -> str:
    pct = min(value / cap * 100, 100) if cap > 0 else 0
    return (
        f'<div class="sbar-wrap">'
        f'<div class="sbar-lbl">{label}: {value:.4f}</div>'
        f'<div class="sbar-track"><div class="sbar-fill {css}" style="width:{pct}%"></div></div>'
        f"</div>"
    )

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_search(client, col, embedder):
    products = _load_products()

    st.markdown('<div class="hero-title">Multi-Aspect Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Search across visual, specification, and review aspects simultaneously</div>',
        unsafe_allow_html=True,
    )

    # --- query + user row ---
    c_q, c_u = st.columns([5, 1])
    with c_q:
        query = st.text_input(
            "query",
            value="Soft breathable linen pants with relaxed fit",
            placeholder="Describe what you're looking for...",
            label_visibility="collapsed",
        )
    with c_u:
        user_id = st.selectbox("user", list(DEMO_USERS), label_visibility="collapsed")

    # --- decomposition preview ---
    if query:
        plan = decompose_query(query)
        st.markdown('<div class="sec-hdr">Query Route Decomposition</div>', unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            t = ", ".join(plan.text_terms) or plan.original_query
            st.markdown(f'<span class="route-badge route-text">Specs: {t}</span>', unsafe_allow_html=True)
        with rc2:
            v = ", ".join(plan.visual_terms) or plan.original_query
            st.markdown(f'<span class="route-badge route-visual">Visual: {v}</span>', unsafe_allow_html=True)
        with rc3:
            r = ", ".join(plan.review_terms) or plan.original_query
            st.markdown(f'<span class="route-badge route-review">Reviews: {r}</span>', unsafe_allow_html=True)

    # --- filters ---
    with st.expander("Filters", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            min_price = st.number_input("Min price ($)", 0.0, step=10.0, value=0.0)
            max_price = st.number_input("Max price ($)", 0.0, step=10.0, value=500.0)
        with f2:
            avail_map = {"Available only": True, "All products": None, "Unavailable only": False}
            avail_label = st.selectbox("Availability", list(avail_map))
        with f3:
            brands = st.multiselect("Brands", sorted({p.brand for p in products}))
            categories = st.multiselect("Categories", sorted({p.category for p in products}))
        with f4:
            colors = st.multiselect("Colors", sorted({p.color for p in products}))
            regions = st.multiselect("Regions", sorted({p.region for p in products}))

    limit = st.slider("Max results", 1, 20, 10)

    # --- search button ---
    if st.button("Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a search query.")
            return

        if not _collection_ok(client, col):
            st.error("Collection not found. Go to **Catalog** and initialize first.")
            return

        filters = SearchFilters(
            min_price=min_price if min_price > 0 else None,
            max_price=max_price if max_price < 500 else None,
            availability=avail_map[avail_label],
            brands=brands or None,
            categories=categories or None,
            regions=regions or None,
            colors=colors or None,
        )
        req = SearchRequest(query=query, user_id=user_id, filters=filters, limit=limit)

        with st.spinner("Searching across all aspects..."):
            try:
                t0 = time.perf_counter()
                results = search_products(client, col, req, embedder)
                elapsed_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                return

        # --- user context ---
        prof = DEMO_USERS.get(user_id)
        if prof:
            st.caption(
                f"**{user_id}** — Brands: {', '.join(prof.preferred_brands)} | "
                f"Price: ${prof.price_range[0]:g}-${prof.price_range[1]:g} | "
                f"Eco: {'Yes' if prof.eco_preference else 'No'}"
            )

        st.markdown(
            f'<div class="sec-hdr">{len(results)} result{"s" if len(results) != 1 else ""} '
            f"in {elapsed_ms:.0f} ms</div>",
            unsafe_allow_html=True,
        )

        if not results:
            st.info("No products matched your query and filters.")
            return

        max_final = max(r.final_score for r in results)

        for rank, res in enumerate(results, 1):
            pay = res.payload
            with st.container(border=True):
                ci, cd, cs = st.columns([1, 3, 2])

                # image
                with ci:
                    img = Path.cwd() / "data" / "images" / f"{res.product_id}.png"
                    if img.exists():
                        st.image(str(img), width=130)

                # details
                with cd:
                    st.markdown(f"**#{rank}  {res.title}**")
                    tags = (
                        f'<span class="tag tag-brand">{res.brand}</span>'
                        f'<span class="tag tag-cat">{res.category}</span>'
                    )
                    if pay.get("availability"):
                        tags += '<span class="tag tag-ok">In Stock</span>'
                    else:
                        tags += '<span class="tag tag-out">Out of Stock</span>'
                    if pay.get("is_sustainable"):
                        tags += '<span class="tag tag-eco">Eco</span>'
                    st.markdown(tags, unsafe_allow_html=True)
                    st.markdown(f'<span class="price">${pay.get("price", 0):.2f}</span>', unsafe_allow_html=True)

                    specs = pay.get("specs", {})
                    if specs:
                        st.caption(" | ".join(f"**{k}**: {v}" for k, v in specs.items()))

                # scores
                with cs:
                    cap = max_final * 1.05
                    html = _score_bar("Qdrant", res.qdrant_score, cap, "sf-qdrant")
                    html += _score_bar("Boost", res.personalization_boost, 1.2, "sf-boost")
                    html += _score_bar("Final", res.final_score, cap, "sf-final")
                    st.markdown(html, unsafe_allow_html=True)

                    with st.expander("Explanation"):
                        for line in res.explanation:
                            marker = "*" if any(k in line for k in ("preferred", "price in", "eco", "favorite")) else "-"
                            st.markdown(f"{marker} {line}")

def page_catalog(client, col, embedder):
    products = _load_products()

    st.markdown('<div class="hero-title">Product Catalog</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Browse the product dataset, initialize collection, and ingest data</div>',
        unsafe_allow_html=True,
    )

    # --- admin row ---
    a1, a2, a3 = st.columns(3)
    with a1:
        profile = st.selectbox("Quantization profile", ["baseline", "scalar", "binary"])
        if st.button("Initialize Collection", use_container_width=True):
            with st.spinner("Creating collection..."):
                try:
                    recreate_collection(client, col, profile=profile)
                    st.success(f"Collection **{col}** created with **{profile}** profile.")
                except Exception as e:
                    st.error(f"Failed: {e}")
    with a2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Ingest Products", use_container_width=True):
            with st.spinner("Embedding and ingesting..."):
                try:
                    ingest_products(client, col, products, embedder)
                    st.success(f"Ingested **{len(products)}** products.")
                except Exception as e:
                    st.error(f"Failed: {e}")
    with a3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    # --- collection status ---
    if _collection_ok(client, col):
        count = _point_count(client, col)
        st.success(f"Collection **{col}** exists — **{count}** points indexed.")
    else:
        st.warning(f"Collection **{col}** not found. Click **Initialize Collection** above.")

    # --- product cards ---
    st.markdown('<div class="sec-hdr">Products</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, p in enumerate(products):
        with cols[idx % 2]:
            with st.container(border=True):
                ic, tc = st.columns([1, 3])
                with ic:
                    if p.image_path and p.image_path.exists():
                        st.image(str(p.image_path), width=120)
                with tc:
                    st.markdown(f"**{p.title}**")
                    st.caption(f"{p.brand} | {p.category} | {p.color} | {p.region}")
                    avail = "Available" if p.availability else "Unavailable"
                    eco = f" | Eco {p.eco_score:.0%}" if p.is_sustainable else ""
                    st.markdown(f"**${p.price:.2f}** {avail}{eco}")

                    with st.expander("Specs and Reviews"):
                        for k, v in p.specs.items():
                            st.markdown(f"- **{k}**: {v}")
                        st.divider()
                        for rev in p.reviews:
                            st.markdown(f'*"{rev}"*')

def page_updates(client, col, embedder):
    products = _load_products()

    st.markdown('<div class="hero-title">Incremental Updates</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Append reviews and images without rebuilding the collection</div>',
        unsafe_allow_html=True,
    )

    if not _collection_ok(client, col):
        st.warning("Collection not found. Go to **Catalog** and initialize first.")
        return

    tab_rev, tab_img = st.tabs(["Add Review", "Add Image"])

    # --- review tab ---
    with tab_rev:
        st.markdown('<div class="sec-hdr">Append a Customer Review</div>', unsafe_allow_html=True)
        pid_r = st.selectbox("Product", [p.id for p in products], key="rev_pid")
        review_text = st.text_area(
            "Review text",
            placeholder="e.g., Amazing fabric quality, feels luxurious on the skin.",
            key="rev_txt",
        )
        if st.button("Submit Review", type="primary", key="rev_btn"):
            if not review_text.strip():
                st.warning("Enter review text.")
            else:
                with st.spinner("Extracting findings, embedding, updating Qdrant..."):
                    try:
                        out = append_review(client, col, pid_r, review_text, embedder)
                        st.success("Review appended!")
                        st.json(out)
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # --- image tab ---
    with tab_img:
        st.markdown('<div class="sec-hdr">Append a Product Image</div>', unsafe_allow_html=True)
        pid_i = st.selectbox("Product", [p.id for p in products], key="img_pid")

        img_dir = Path.cwd() / "data" / "images"
        images = sorted(img_dir.glob("*.png")) if img_dir.exists() else []

        if not images:
            st.info("No images found. Run **Catalog > Ingest** first to generate product images.")
            return

        chosen = st.selectbox("Image file", images, format_func=lambda p: p.name, key="img_sel")
        if chosen:
            st.image(str(chosen), width=200, caption=chosen.name)

        if st.button("Append Image", type="primary", key="img_btn"):
            with st.spinner("Generating patch embeddings, updating Qdrant..."):
                try:
                    out = append_image(client, col, pid_i, chosen, embedder)
                    st.success("Image patches appended!")
                    st.json(out)
                except Exception as e:
                    st.error(f"Failed: {e}")


def page_benchmarks(client, col, embedder):
    st.markdown('<div class="hero-title">Performance Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Compare quantization profiles: latency, recall, and estimated storage</div>',
        unsafe_allow_html=True,
    )

    profiles = st.multiselect(
        "Profiles to benchmark",
        ["baseline", "scalar", "binary", "hnsw"],
        default=["baseline", "scalar", "binary"],
    )

    if st.button("Run Benchmarks", type="primary", use_container_width=True):
        if not profiles:
            st.warning("Select at least one profile.")
            return

        all_results: list[dict] = []
        bar = st.progress(0, text="Starting...")

        for i, prof in enumerate(profiles):
            bar.progress(i / len(profiles), text=f"Benchmarking **{prof}**...")
            try:
                res = run_benchmark(client, col, Path.cwd(), embedder, prof)
                all_results.append(result_dict(res))
            except Exception as e:
                st.error(f"`{prof}` failed: {e}")

        bar.progress(1.0, text="Done!")

        if not all_results:
            return

        # --- metric cards ---
        st.markdown('<div class="sec-hdr">Results</div>', unsafe_allow_html=True)

        for r in all_results:
            with st.container(border=True):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Profile", r["profile"].upper())
                m2.metric("Mean Latency", f'{r["mean_latency_ms"]:.1f} ms')
                m3.metric("P95 Latency", f'{r["p95_latency_ms"]:.1f} ms')
                m4.metric("Recall@3", f'{r["recall_at_3"]:.0%}')
                m5.metric("Est. Storage", f'{r["storage_size_bytes"] / 1024:.1f} KB')

        # --- comparison table ---
        st.markdown('<div class="sec-hdr">Detailed Comparison</div>', unsafe_allow_html=True)
        st.dataframe(all_results, use_container_width=True, hide_index=True)

        # --- charts (only with 2+ profiles) ---
        if len(all_results) >= 2:
            import pandas as pd

            df = pd.DataFrame(all_results)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Mean Latency (ms)** — lower is better")
                st.bar_chart(df.set_index("profile")["mean_latency_ms"])
            with c2:
                st.markdown("**Estimated Storage (bytes)** — lower is better")
                st.bar_chart(df.set_index("profile")["storage_size_bytes"])

def main():
    # ---- sidebar ----
    with st.sidebar:
        st.markdown(
            '<div class="hero-title" style="font-size:1.35rem">Commerce Engine</div>',
            unsafe_allow_html=True,
        )
        st.caption("Multi-Aspect Semantic Search | By Divy Yadav")
        st.divider()

        qdrant_url = st.text_input("Qdrant URL", value="http://localhost:6333")
        collection = st.text_input("Collection", value="commerce_products")
        backend = st.selectbox("Embedding backend", ["deterministic", "production"])

        if backend == "production":
            st.caption("Production mode downloads ML models on first use.")

        st.divider()

        page = st.radio(
            "nav",
            ["Search", "Catalog", "Updates", "Benchmarks"],
            label_visibility="collapsed",
        )

        st.divider()

        # connection status
        client = _make_client(qdrant_url)
        embedder = _make_embedder(backend)
        connected = _qdrant_ok(client)

        if connected:
            if getattr(client, "is_memory_fallback", False):
                st.markdown(
                    '<span class="dot dot-on"></span> Qdrant connected (in-memory)', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="dot dot-on"></span> Qdrant connected', unsafe_allow_html=True
                )

            if _collection_ok(client, collection):
                n = _point_count(client, collection)
                st.caption(f"`{collection}` — {n} points")
            else:
                st.caption(f"`{collection}` not found")
        else:
            st.markdown(
                '<span class="dot dot-off"></span> Qdrant disconnected', unsafe_allow_html=True
            )
            st.caption("Start: `docker compose up -d qdrant`")

    # ---- main area ----
    if not connected:
        st.markdown('<div class="hero-title">Multi-Aspect Commerce Engine</div>', unsafe_allow_html=True)
        st.error("**Cannot connect to Qdrant.** Start the server first:")
        st.code("docker compose up -d qdrant", language="bash")
        st.info("Then refresh this page.")
        return

    if page == "Search":
        page_search(client, collection, embedder)
    elif page == "Catalog":
        page_catalog(client, collection, embedder)
    elif page == "Updates":
        page_updates(client, collection, embedder)
    elif page == "Benchmarks":
        page_benchmarks(client, collection, embedder)


if __name__ == "__main__":
    main()
