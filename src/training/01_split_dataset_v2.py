"""
Pregatirea datasetului v2 pentru antrenare: split stratificat 70/15/15.

Pasi:
1. Incarca dataset_licenta_complet.csv (1483 articole, schema canonica 18 coloane)
2. Construieste coloana `text` = titlu + "\\n\\n" + stire_citata (input clasificator)
   - IMPORTANT: NU folosim text_curat, care pe Veridica contine demontarea jurnalistului.
     Input corect pentru clasificare este naratiunea pro-Kremlin izolata (stire_citata).
3. Stratificare pe (label_numeric, sursa_site) pentru a pastra distributia in toate split-urile
4. Salveaza train/val/test + metadata JSON

Usage:
    python src/training/01_split_dataset_v2.py \
        --input data/raw/dataset_licenta_complet.csv \
        --outdir data/processed
"""
import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def construieste_text_input(row: pd.Series) -> str:
    """Construieste input-ul clasificatorului: titlu + stire_citata.

    Motivatie: pe Veridica.ro, `text_curat` contine demontarea facuta de jurnalist
    (deci ar contamina clasa 1 cu voce de fact-checker). Coloana `stire_citata`
    izoleaza citatul pro-Kremlin propriu-zis.
    """
    titlu = str(row["titlu"]).strip()
    stire = str(row["stire_citata"]).strip()
    return f"{titlu}\n\n{stire}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path către dataset_licenta_complet.csv")
    parser.add_argument("--outdir", default="data/processed", help="Director output")
    parser.add_argument("--seed", type=int, default=42, help="Seed pentru reproducibilitate")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # === 1. Incarcare ===
    print(f"[INFO] Încărcare: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[INFO] Shape inițial: {df.shape}")

    # === 2. Verificari integritate ===
    assert df["label_numeric"].isin([0, 1]).all(), "label_numeric trebuie să fie 0 sau 1"
    assert df["titlu"].notna().all(), "NaN în coloana titlu"
    assert df["stire_citata"].notna().all(), "NaN în coloana stire_citata"
    assert df["sursa_site"].notna().all(), "NaN în coloana sursa_site"

    # Verificare duplicate pe hash_continut
    n_dup = df["hash_continut"].duplicated().sum()
    if n_dup > 0:
        print(f"[WARN] {n_dup} duplicate pe hash_continut — elimin")
        df = df.drop_duplicates(subset=["hash_continut"]).reset_index(drop=True)

    # === 3. Construire coloana `text` (input clasificator) ===
    df["text"] = df.apply(construieste_text_input, axis=1)
    df["nr_cuvinte_text"] = df["text"].str.split().str.len()
    print(f"[INFO] Statistici lungime text (cuvinte): "
          f"min={df['nr_cuvinte_text'].min()}, "
          f"median={int(df['nr_cuvinte_text'].median())}, "
          f"max={df['nr_cuvinte_text'].max()}")

    # === 4. Split stratificat pe (label_numeric, sursa_site) ===
    # Cream cheie de stratificare combinata
    df["strat_key"] = df["label_numeric"].astype(str) + "_" + df["sursa_site"]
    print(f"\n[INFO] Distribuție strat_key:")
    print(df["strat_key"].value_counts().to_string())

    # Split initial: 70% train, 30% temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["strat_key"],
        random_state=args.seed,
    )

    # Split temp in val/test: 50/50 → 15%/15% din total
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["strat_key"],
        random_state=args.seed,
    )

    # === 5. Verificari post-split ===
    print(f"\n[INFO] Dimensiuni split:")
    print(f"  TRAIN: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    print(f"  VAL:   {len(val_df)} ({100*len(val_df)/len(df):.1f}%)")
    print(f"  TEST:  {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")

    for name, d in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n  [{name}] distribuție label_numeric:")
        print("    " + d["label_numeric"].value_counts().to_string().replace("\n", "\n    "))
        print(f"  [{name}] distribuție sursa_site:")
        print("    " + d["sursa_site"].value_counts().to_string().replace("\n", "\n    "))

    # Zero overlap pe id
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])
    assert len(train_ids & val_ids) == 0, "Overlap TRAIN/VAL"
    assert len(train_ids & test_ids) == 0, "Overlap TRAIN/TEST"
    assert len(val_ids & test_ids) == 0, "Overlap VAL/TEST"
    print("\n[OK] Zero overlap pe id între split-uri")

    # === 6. Salvare ===
    # Drop coloane auxiliare inainte de salvare
    cols_drop = ["strat_key"]
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d_out = d.drop(columns=cols_drop).reset_index(drop=True)
        path = outdir / f"dataset_v2_{name}.csv"
        d_out.to_csv(path, index=False)
        print(f"[OK] Salvat: {path} ({len(d_out)} rânduri)")

    # Metadata
    meta = {
        "versiune_dataset": "v2",
        "seed": args.seed,
        "input_path": str(args.input),
        "n_total": len(df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "distributie_clase": {
            "cls0_credibil": int((df["label_numeric"] == 0).sum()),
            "cls1_dezinformare": int((df["label_numeric"] == 1).sum()),
        },
        "distributie_surse": df["sursa_site"].value_counts().to_dict(),
        "stratificare": "label_numeric + sursa_site",
        "coloana_text": "titlu + \\n\\n + stire_citata",
        "diferente_fata_de_v1": [
            "volum 1483 (vs 1427 v1)",
            "include 2022 (lipsă complet în v1)",
            "Stopfals.md ca sursă nouă clasa 1 (85 articole)",
            "entity balancing Moldova aplicat (DF diff redus de la +30.8% la +16.1%)",
        ],
    }
    meta_path = outdir / "dataset_v2_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
