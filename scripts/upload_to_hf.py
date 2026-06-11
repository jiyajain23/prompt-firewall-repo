"""
Upload all trained artefacts to a HuggingFace model repository.

Usage:
    python scripts/upload_to_hf.py \
        --repo-id your-username/prompt-firewall \
        --model-dir /path/to/bouncer_model

Required HF token: set HF_TOKEN env var or run `huggingface-cli login` first.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ARTEFACTS = [
    "xgb_model.json",
    "scaler.pkl",
    "xgb_threshold.pkl",
    "ensemble_config.pkl",
    "zeroday.faiss",
    "zeroday_meta.pkl",
    "shap_explainer.pkl",
    "taxonomy_clf.pkl",
    "taxonomy_le.pkl",
]

DEBERTA_DIR = "deberta_best"


def upload(repo_id: str, model_dir: str, private: bool = True) -> None:
    api   = HfApi()
    token = os.environ.get("HF_TOKEN")

    print(f"Creating repo {repo_id} (private={private})...")
    create_repo(repo_id, repo_type="model", private=private,
                token=token, exist_ok=True)

    d = Path(model_dir)

    # Upload flat artefacts
    for fname in ARTEFACTS:
        fpath = d / fname
        if not fpath.exists():
            print(f"   ⚠️  {fname} not found, skipping")
            continue
        print(f"   Uploading {fname} ({fpath.stat().st_size / 1024:.1f} KB)...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    # Upload DeBERTa checkpoint directory
    deberta_path = d / DEBERTA_DIR
    if deberta_path.exists():
        print(f"   Uploading {DEBERTA_DIR}/ ...")
        api.upload_folder(
            folder_path=str(deberta_path),
            path_in_repo=DEBERTA_DIR,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
    else:
        print(f"   ⚠️  {DEBERTA_DIR}/ not found")

    # Upload configs
    print("   Uploading configs/...")
    api.upload_folder(
        folder_path="configs",
        path_in_repo="configs",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    print(f"\n✅ Upload complete → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id",   required=True, help="HF repo, e.g. user/prompt-firewall")
    parser.add_argument("--model-dir", required=True, help="Local artefacts directory")
    parser.add_argument("--public",    action="store_true", help="Make repo public (default: private)")
    args = parser.parse_args()
    upload(args.repo_id, args.model_dir, private=not args.public)
