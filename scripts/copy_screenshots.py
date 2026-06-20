#!/usr/bin/env python3
"""Script to copy generated screenshots from the brain directory to the docs/images directory.

Since the terminal sandbox is isolated, run this script on your host machine to copy the screenshots:
    python scripts/copy_screenshots.py
"""

import shutil
from pathlib import Path

def main():
    home = Path.home()
    # Find the most recent conversation folder under brain
    brain_root = home / ".gemini" / "antigravity-ide" / "brain"
    
    if not brain_root.exists():
        print(f"Error: Brain directory not found at {brain_root}")
        return
        
    # Get the latest modified conversation directory
    conv_dirs = [d for d in brain_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not conv_dirs:
        print("Error: No conversation directories found.")
        return
        
    latest_conv = max(conv_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Using latest conversation directory: {latest_conv.name}")

    dest_dir = Path(__file__).parent.parent / "docs" / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Find the screenshots matching our patterns
    mapping = {
        "fastapi_swagger": "fastapi_swagger.png",
        "search_page": "search_page.png",
        "search_results": "search_results.png",
        "product_catalog": "product_catalog.png",
        "update_workflow": "update_workflow.png",
        "benchmark_dashboard": "benchmark_dashboard.png",
    }

    copied_count = 0
    for file in latest_conv.iterdir():
        if file.suffix == ".png":
            for prefix, dest_name in mapping.items():
                if file.name.startswith(prefix):
                    dest_path = dest_dir / dest_name
                    print(f"Copying {file.name} -> docs/images/{dest_name}")
                    shutil.copy2(file, dest_path)
                    copied_count += 1
                    break
                    
    print(f"\nSuccessfully copied {copied_count} screenshots to docs/images/")

if __name__ == "__main__":
    main()
