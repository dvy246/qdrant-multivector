import shutil
from pathlib import Path

p1 = ["", "Users", "divyyadav", ".gemini", "antigravity-ide", "brain", "8ee3e1d7-685c-44bf-87fd-08cb7f127f66"]
p2 = ["", "Users", "divyyadav", "Desktop", "qdrant-multivector", "docs", "images"]

brain_dir = Path("/".join(p1))
dest_dir = Path("/".join(p2))

mapping = {
    "fastapi_swagger_1781943583199.png": "fastapi_swagger.png",
    "search_page_1781943625468.png": "search_page.png",
    "search_results_1781943682240.png": "search_results.png",
    "product_catalog_1781943704700.png": "product_catalog.png",
    "update_workflow_1781943722767.png": "update_workflow.png",
    "benchmark_dashboard_1781943746573.png": "benchmark_dashboard.png",
}

dest_dir.mkdir(parents=True, exist_ok=True)

for src_name, dest_name in mapping.items():
    src_path = brain_dir / src_name
    dest_path = dest_dir / dest_name
    print(f"Copying {src_name} -> {dest_name}")
    try:
        shutil.copy2(src_path, dest_path)
        print("Success")
    except Exception as e:
        print(f"Failed: {e}")
