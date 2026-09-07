"""Package BETA repository into a standalone bundle for Kaggle execution."""
from __future__ import annotations

import base64
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = REPO_ROOT / "experiments" / "beta_bundle.tar.gz"
SNIPPET_PATH = REPO_ROOT / "experiments" / "kaggle_unpacker_cell.py"


def make_tarball(output_path: Path):
    items_to_pack = ["beta", "configs", "assets", "experiments"]
    with tarfile.open(output_path, "w:gz") as tar:
        for item in items_to_pack:
            p = REPO_ROOT / item
            if p.exists():
                print(f"Adding {item}/ to archive...")
                tar.add(p, arcname=item)
    print(f"Created bundle: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def make_unpacker_snippet(bundle_path: Path, snippet_path: Path):
    b64_data = base64.b64encode(bundle_path.read_bytes()).decode("utf-8")
    code = f'''# --- Self-Contained BETA Unpacker (Paste into Kaggle notebook cell) ---
import base64
import io
import tarfile
from pathlib import Path

BUNDLE_B64 = """{b64_data}"""

print("Unpacking BETA codebase and assets into current workspace...")
tar_bytes = base64.b64decode(BUNDLE_B64)
with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
    tar.extractall(path="/kaggle/working" if Path("/kaggle/working").exists() else ".")
print("BETA codebase, assets, and configs unpacked successfully!")
'''
    snippet_path.write_text(code)
    print(f"Generated unpacker snippet: {snippet_path} ({snippet_path.stat().st_size / 1024:.1f} KB)")


def main():
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_tarball(BUNDLE_PATH)
    make_unpacker_snippet(BUNDLE_PATH, SNIPPET_PATH)
    print("\nDone! You can:")
    print("1. Upload experiments/beta_bundle.tar.gz as a Kaggle Dataset, OR")
    print("2. Use the Python code in experiments/kaggle_unpacker_cell.py directly in a Kaggle notebook cell.")


if __name__ == "__main__":
    main()
