"""
Urca modelul antrenat si dataset-ul pe Hugging Face Hub.

Inainte de rulare:
    source .venv/bin/activate
    hf auth login   (token Write de pe huggingface.co/settings/tokens)

Rulare:
    python scripts/upload_hf.py
    python scripts/upload_hf.py --doar-dataset
    python scripts/upload_hf.py --doar-model
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_REPO   = "rares127/xlmr-dezinformare-ro"
DATASET_REPO = "rares127/dezinformare-ro"

MODEL_DIR    = ROOT / "models" / "xlmr_baseline_v2" / "final"
DATASET_CSV  = ROOT / "data" / "final" / "dataset_licenta_complet.csv"
MODEL_CARD   = ROOT / "hf_cards" / "model_card.md"
DATASET_CARD = ROOT / "hf_cards" / "dataset_card.md"


def verifica_dependente():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        print("Eroare: pip install huggingface_hub")
        sys.exit(1)


def upload_model():
    from huggingface_hub import HfApi, create_repo

    if not MODEL_DIR.exists():
        print(f"Eroare: {MODEL_DIR} nu exista. Antreneaza modelul intai.")
        sys.exit(1)
    if not MODEL_CARD.exists():
        print(f"Eroare: {MODEL_CARD} lipsa.")
        sys.exit(1)

    api = HfApi()
    print(f"Creare repo model: {MODEL_REPO}")
    create_repo(MODEL_REPO, repo_type="model", exist_ok=True)

    # incarca cardul ca README.md
    print("Upload model card...")
    api.upload_file(
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        repo_id=MODEL_REPO,
        repo_type="model",
    )

    # incarca fisierele modelului (safetensors + tokenizer + config)
    print(f"Upload model files din {MODEL_DIR} (~1.1 GB, poate dura 3-5 minute)...")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=MODEL_REPO,
        repo_type="model",
        ignore_patterns=["README.md"],  # cardul e deja incarcat
    )
    print(f"Model disponibil la: https://huggingface.co/{MODEL_REPO}")


def upload_dataset():
    from huggingface_hub import HfApi, create_repo

    if not DATASET_CSV.exists():
        print(f"Eroare: {DATASET_CSV} nu exista.")
        sys.exit(1)
    if not DATASET_CARD.exists():
        print(f"Eroare: {DATASET_CARD} lipsa.")
        sys.exit(1)

    api = HfApi()
    print(f"Creare repo dataset: {DATASET_REPO}")
    create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)

    # incarca cardul ca README.md
    print("Upload dataset card...")
    api.upload_file(
        path_or_fileobj=str(DATASET_CARD),
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )

    # incarca CSV-ul (~5.7 MB)
    print("Upload dataset CSV (~5.7 MB)...")
    api.upload_file(
        path_or_fileobj=str(DATASET_CSV),
        path_in_repo="dataset_licenta_complet.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Dataset disponibil la: https://huggingface.co/datasets/{DATASET_REPO}")


def main():
    parser = argparse.ArgumentParser(description="Upload model + dataset pe Hugging Face")
    parser.add_argument("--doar-model",   action="store_true")
    parser.add_argument("--doar-dataset", action="store_true")
    args = parser.parse_args()

    verifica_dependente()

    if args.doar_model:
        upload_model()
    elif args.doar_dataset:
        upload_dataset()
    else:
        upload_dataset()   # dataset-ul e mai mic, merge primul
        print()
        upload_model()


if __name__ == "__main__":
    main()
