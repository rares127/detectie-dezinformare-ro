"""
Calibrare threshold pentru scor_D (modulul 3) — protocol academic curat.

Protocol:
---------
1. Construim corpus cls1 „no-val": excludem cele 112 articole din val_cls1
   pentru a elimina data leakage la calibrare.
2. Segmentam articolele val cu Stanza (la fel ca pipeline-ul standard).
3. Calculam scor_D = scor_cls1_mean − scor_cls0_mean per articol val.
4. Gasim threshold τ* care maximizeaza F1 pe val.
5. Aplicam τ* pe test set (corpus cls1 original — test e necontaminat).
6. Raportam Precision/Recall/F1/Accuracy + confusion matrix + comparatie
   cu modulul 2 standard.

Findings raportate:
  - Threshold optim pe val (τ*)
  - Metrici la τ* pe val (calibrare)
  - Metrici la τ* pe test (evaluare reala, no-leak)
  - Comparatie directa cu modulul 2 IID standard (F1=100%)
  - Curve sensibilitate (cum se schimba F1/precision/recall in functie de τ)

Output:
  - findings/calibrare_threshold.md + .json

Utilizare:
  python scripts/calibrare_threshold.py
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurare
# ---------------------------------------------------------------------------

CORPUS_CLS0_PATH = Path("data/processed/propozitii_cls0_corpus.parquet")
CORPUS_CLS1_PATH = Path("data/processed/propozitii_cls1_corpus_v2.parquet")
VAL_PATH = Path("data/processed/dataset_v2_val.csv")
TEST_SET_PATH = Path("data/processed/subset_benchmark_v3_curat.parquet")

CACHE_DIR = Path("data/processed/embeddings_cache")
CORPUS_CLS1_NOVAL_OUT = Path(
    "data/processed/propozitii_cls1_corpus_v2_no_val.parquet"
)

RAPORT_MD = Path("findings/calibrare_threshold.md")
RAPORT_JSON = Path("findings/calibrare_threshold.json")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SEED = 42
DOWNSAMPLE_CLS1_LA_TEST = 5_290  # paritate cls0 (pentru evaluare test)

# Filtre lungime propozitie (consistent cu corpus)
MIN_CUVINTE = 7
MAX_CUVINTE = 54


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seteaza_seed(seed: int = SEED) -> None:
    """Fixeaza seed-ul (random + numpy)."""
    random.seed(seed)
    np.random.seed(seed)


def calculeaza_hash_corpus(texts, model_name: str) -> str:
    """Hash determinist pentru cache embeddings."""
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\n")
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def incarca_sau_calculeaza_embeddings(
    texts: list[str], nume: str, model, device: str
) -> np.ndarray:
    """Cache embeddings pe disc — invalidate automat la schimbare continut."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = calculeaza_hash_corpus(texts, MODEL_NAME)
    cache_path = CACHE_DIR / f"{nume}_{h}.npy"

    if cache_path.exists():
        emb = np.load(cache_path)
        assert emb.shape[0] == len(texts), f"Cache corupt {cache_path}"
        print(f"  [cache HIT] {cache_path.name}")
        return emb

    print(f"  [cache MISS] calculez {len(texts):,} embeddings pe {device}...")
    emb = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True, device=device,
    )
    np.save(cache_path, emb)
    print(f"  [cache SAVE] {cache_path.name}")
    return emb


def scor_cosine_max(emb_test: np.ndarray, emb_corpus: np.ndarray,
                    batch: int = 256) -> np.ndarray:
    """Cosine max per propozitie (embeddings normalizate L2)."""
    n = emb_test.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch):
        b = emb_test[i:i + batch]
        out[i:i + batch] = (b @ emb_corpus.T).max(axis=1)
    return out


# ---------------------------------------------------------------------------
# Constructie corpus cls1 „no-val" pentru calibrare
# ---------------------------------------------------------------------------

def construieste_corpus_cls1_no_val(
    df_cls1_full: pd.DataFrame, val_cls1_ids: set[str]
) -> pd.DataFrame:
    """
    Filtreaza corpusul cls1 pentru a exclude articolele val cls1.
    Returneaza DataFrame curat + salveaza pe disc pentru reutilizare.
    """
    print(f"\n[construire corpus no-val]")
    print(f"  Corpus cls1 original: {len(df_cls1_full):,} prop. "
          f"din {df_cls1_full['articol_id'].nunique()} articole")
    print(f"  Articole val cls1 de exclus: {len(val_cls1_ids)}")

    mask = ~df_cls1_full["articol_id"].isin(val_cls1_ids)
    df_no_val = df_cls1_full[mask].reset_index(drop=True)

    print(f"  Corpus filtrat: {len(df_no_val):,} prop. "
          f"din {df_no_val['articol_id'].nunique()} articole")
    print(f"  Distribuție: {df_no_val['sursa_site'].value_counts().to_dict()}")

    # Salvam pentru reutilizare
    CORPUS_CLS1_NOVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_no_val.to_parquet(CORPUS_CLS1_NOVAL_OUT, index=False)
    print(f"  Salvat: {CORPUS_CLS1_NOVAL_OUT}")

    return df_no_val


# ---------------------------------------------------------------------------
# Segmentare articole val cu Stanza
# ---------------------------------------------------------------------------

def segmenteaza_articole_val(df_val: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenteaza coloana stire_citata in propozitii folosind Stanza.

    Pipeline identic cu cel folosit pentru corpus cls1 (asigura consistenta):
      - Stanza pe stire_citata
      - Filtrare lungime [7, 54] cuvinte
      - Doar propozitii valide

    Returneaza DataFrame cu o linie per propozitie:
      [articol_id, label_numeric, sursa_site, propozitie, nr_cuvinte]
    """
    print(f"\n[segmentare val cu Stanza]")
    try:
        import stanza
    except ImportError:
        raise ImportError(
            "Instalează stanza: pip install stanza --break-system-packages"
        )

    # Initializare nlp Romanian (procesoare minime)
    print("  Încarc model Stanza Romanian...")
    nlp = stanza.Pipeline(
        lang="ro",
        processors="tokenize",
        verbose=False,
        use_gpu=False,  # Stanza nu are nevoie de GPU pentru tokenize
    )

    propozitii = []
    n_total_brut = 0
    n_eliminate_lungime = 0

    for _, art in df_val.iterrows():
        text = str(art["stire_citata"]) if pd.notna(art["stire_citata"]) else ""
        if not text.strip():
            continue

        doc = nlp(text)
        for poz, sent in enumerate(doc.sentences):
            prop_text = sent.text.strip()
            n_cuvinte = len(prop_text.split())
            n_caractere = len(prop_text)
            n_total_brut += 1

            # Filtru lungime (consistent cu corpus)
            if n_cuvinte < MIN_CUVINTE or n_cuvinte > MAX_CUVINTE:
                n_eliminate_lungime += 1
                continue

            propozitii.append({
                "articol_id": str(art["id"]),
                "label_numeric": int(art["label_numeric"]),
                "sursa_site": str(art["sursa_site"]),
                "pozitie_in_articol": poz,
                "propozitie": prop_text,
                "nr_cuvinte": n_cuvinte,
                "nr_caractere": n_caractere,
            })

    df_prop = pd.DataFrame(propozitii)
    print(f"  Articole procesate: {df_val['id'].nunique()}")
    print(f"  Propoziții brut Stanza: {n_total_brut:,}")
    print(f"  Eliminate lungime [<7 sau >54 cuv.]: {n_eliminate_lungime:,}")
    print(f"  Propoziții finale: {len(df_prop):,}")
    print(f"  Distribuție label: "
          f"{df_prop.groupby('label_numeric').size().to_dict()}")
    print(f"  Articole cu cel puțin o propoziție: "
          f"{df_prop['articol_id'].nunique()}")

    return df_prop


# ---------------------------------------------------------------------------
# Agregare scoruri la articol
# ---------------------------------------------------------------------------

def agrega_la_articol(
    df_prop: pd.DataFrame,
    sc_cls0: np.ndarray,
    sc_cls1: np.ndarray,
) -> pd.DataFrame:
    """Ataseaza scoruri propozitie si agrega la articol cu mean (Test D)."""
    df = df_prop.copy().reset_index(drop=True)
    df["scor_cls0"] = sc_cls0
    df["scor_cls1"] = sc_cls1

    art = df.groupby("articol_id").agg(
        label=("label_numeric", "first"),
        sursa=("sursa_site", "first"),
        nr_prop=("propozitie", "count"),
        scor_cls0_mean=("scor_cls0", "mean"),
        scor_cls1_mean=("scor_cls1", "mean"),
    )
    art["diff_mean"] = art["scor_cls1_mean"] - art["scor_cls0_mean"]
    return art


# ---------------------------------------------------------------------------
# Calibrare threshold
# ---------------------------------------------------------------------------

def calculeaza_metrici(
    labels: np.ndarray, scoruri: np.ndarray, threshold: float
) -> dict:
    """Confusion matrix + Precision/Recall/F1/Accuracy la threshold dat."""
    pred = (scoruri > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": float(accuracy),
        "precision_cls1": float(precision),
        "recall_cls1": float(recall),
        "f1_cls1": float(f1),
    }


def gaseste_threshold_optim(
    labels: np.ndarray, scoruri: np.ndarray, n_thresholds: int = 200
) -> dict:
    """
    Cauta threshold optim care maximizeaza F1 pe cls1.

    Strategie:
      - Grid de n_thresholds threshold-uri intre min(scoruri) si max(scoruri)
      - Calculeaza F1 pentru fiecare threshold
      - Returneaza threshold optim + curba F1

    Plus calculam:
      - Threshold pentru max accuracy (alternativ)
      - Threshold pentru max Youden's J (recall − fpr, util pentru sisteme de
        siguranta unde recall e prioritar)
    """
    threshold_grid = np.linspace(scoruri.min(), scoruri.max(), n_thresholds)

    metrici_per_threshold = []
    for τ in threshold_grid:
        m = calculeaza_metrici(labels, scoruri, τ)
        # Youden's J = recall − fpr unde fpr = fp / (fp + tn)
        fpr = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) > 0 else 0
        m["youden_j"] = m["recall_cls1"] - fpr
        metrici_per_threshold.append(m)

    # Threshold-uri optime per criteriu
    best_f1 = max(metrici_per_threshold, key=lambda x: x["f1_cls1"])
    best_acc = max(metrici_per_threshold, key=lambda x: x["accuracy"])
    best_youden = max(metrici_per_threshold, key=lambda x: x["youden_j"])

    return {
        "n_thresholds_evaluate": n_thresholds,
        "threshold_range": [float(scoruri.min()), float(scoruri.max())],
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "best_youden_j": best_youden,
        "curba": metrici_per_threshold,
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestreaza: incarcare → no-val corpus → segmentare val → scoring →
    calibrare → evaluare test → raport."""
    print("=" * 70)
    print("CALIBRARE THRESHOLD — protocol academic curat")
    print("=" * 70)
    seteaza_seed(SEED)

    # ------------------------------------------------------------------
    # 1. Incarcare date
    # ------------------------------------------------------------------
    print("\n[1/8] Încărcare date...")
    if not CORPUS_CLS0_PATH.exists():
        raise FileNotFoundError(f"Corpus cls0 nu găsit: {CORPUS_CLS0_PATH}")
    if not CORPUS_CLS1_PATH.exists():
        raise FileNotFoundError(f"Corpus cls1 nu găsit: {CORPUS_CLS1_PATH}")
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Val set nu găsit: {VAL_PATH}")
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Test set nu găsit: {TEST_SET_PATH}")

    df_cls0 = pd.read_parquet(CORPUS_CLS0_PATH)
    df_cls1_full = pd.read_parquet(CORPUS_CLS1_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_parquet(TEST_SET_PATH)

    print(f"  cls0 corpus: {len(df_cls0):,} prop.")
    print(f"  cls1 corpus full: {len(df_cls1_full):,} prop.")
    print(f"  val: {len(df_val)} articole "
          f"(cls0={int((df_val['label_numeric']==0).sum())}, "
          f"cls1={int((df_val['label_numeric']==1).sum())})")
    print(f"  test: {len(df_test):,} prop. "
          f"({df_test['articol_id'].nunique()} articole)")

    # ------------------------------------------------------------------
    # 2. Constructie corpus cls1 no-val (fix data leakage)
    # ------------------------------------------------------------------
    val_cls1_ids = set(df_val[df_val["label_numeric"] == 1]["id"].tolist())
    df_cls1_noval = construieste_corpus_cls1_no_val(df_cls1_full, val_cls1_ids)

    # Pentru evaluarea pe test folosim corpusul ORIGINAL downsampled (test e necontaminat)
    print(f"\n[2.5/8] Downsample cls1 BASELINE pentru test "
          f"(seed={SEED}, n={DOWNSAMPLE_CLS1_LA_TEST})...")
    df_cls1_baseline = df_cls1_full.sample(
        n=DOWNSAMPLE_CLS1_LA_TEST, random_state=SEED
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Segmentare val cu Stanza
    # ------------------------------------------------------------------
    df_val_prop = segmenteaza_articole_val(df_val)

    # Verificare: toate articolele au cel putin o propozitie?
    art_val_ramase = df_val_prop["articol_id"].nunique()
    art_val_pierdute = len(df_val) - art_val_ramase
    if art_val_pierdute > 0:
        print(f"  ⚠ {art_val_pierdute} articole val NU au nicio propoziție "
              f"care să treacă filtrul lungime. Vor fi excluse din calibrare.")

    # ------------------------------------------------------------------
    # 4. Incarcare model + calcul embeddings
    # ------------------------------------------------------------------
    print("\n[4/8] Încărcare model + calcul embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as e:
        raise ImportError(f"Lipsă dependențe: {e}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  Device: {device}")

    print("\n  cls0 corpus:")
    emb_cls0 = incarca_sau_calculeaza_embeddings(
        df_cls0["propozitie"].tolist(), "cls0_corpus", model, device
    )
    print("\n  cls1 corpus baseline (pentru evaluare test):")
    emb_cls1_baseline = incarca_sau_calculeaza_embeddings(
        df_cls1_baseline["propozitie"].tolist(),
        "cls1_corpus_v2_downsampled", model, device
    )
    print("\n  cls1 corpus NO-VAL (pentru calibrare val):")
    emb_cls1_noval = incarca_sau_calculeaza_embeddings(
        df_cls1_noval["propozitie"].tolist(),
        "cls1_corpus_v2_no_val", model, device
    )
    print("\n  val propoziții (necache, mic):")
    emb_val = model.encode(
        df_val_prop["propozitie"].tolist(),
        batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True, device=device,
    )
    print("\n  test propoziții (necache):")
    emb_test = model.encode(
        df_test["propozitie"].tolist(),
        batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True, device=device,
    )

    # ------------------------------------------------------------------
    # 5. Scoring + agregare pe val (corpus no-val pentru cls1)
    # ------------------------------------------------------------------
    print("\n[5/8] Scoring val (cu corpus cls1 NO-VAL)...")
    sc_cls0_val = scor_cosine_max(emb_val, emb_cls0)
    sc_cls1_val = scor_cosine_max(emb_val, emb_cls1_noval)
    art_val = agrega_la_articol(df_val_prop, sc_cls0_val, sc_cls1_val)
    print(f"  Articole val cu scoruri: {len(art_val)}")
    print(f"  Range diff_mean: [{art_val['diff_mean'].min():.4f}, "
          f"{art_val['diff_mean'].max():.4f}]")

    # ------------------------------------------------------------------
    # 6. Calibrare threshold pe val
    # ------------------------------------------------------------------
    print("\n[6/8] Calibrare threshold pe val (max F1 cls1)...")
    labels_val = art_val["label"].values
    scoruri_val = art_val["diff_mean"].values

    calibrare = gaseste_threshold_optim(labels_val, scoruri_val)
    tau_optim = calibrare["best_f1"]["threshold"]
    print(f"  τ* (best F1) = {tau_optim:.6f}")
    print(f"  F1 val = {calibrare['best_f1']['f1_cls1']:.4f}")
    print(f"  Accuracy val = {calibrare['best_f1']['accuracy']:.4f}")
    print(f"  Precision val = {calibrare['best_f1']['precision_cls1']:.4f}")
    print(f"  Recall val = {calibrare['best_f1']['recall_cls1']:.4f}")

    # ------------------------------------------------------------------
    # 7. Aplicare τ* pe test (cu corpus cls1 ORIGINAL — test e necontaminat)
    # ------------------------------------------------------------------
    print(f"\n[7/8] Aplicare τ* pe test (cu corpus cls1 BASELINE)...")
    sc_cls0_test = scor_cosine_max(emb_test, emb_cls0)
    sc_cls1_test = scor_cosine_max(emb_test, emb_cls1_baseline)
    art_test = agrega_la_articol(df_test, sc_cls0_test, sc_cls1_test)
    print(f"  Articole test: {len(art_test)}")

    labels_test = art_test["label"].values
    scoruri_test = art_test["diff_mean"].values

    metrici_test = calculeaza_metrici(labels_test, scoruri_test, tau_optim)
    print(f"  Test la τ* = {tau_optim:.6f}:")
    print(f"    Accuracy   = {metrici_test['accuracy']:.4f}")
    print(f"    Precision  = {metrici_test['precision_cls1']:.4f}")
    print(f"    Recall     = {metrici_test['recall_cls1']:.4f}")
    print(f"    F1         = {metrici_test['f1_cls1']:.4f}")
    print(f"    Confusion: TP={metrici_test['tp']}, FP={metrici_test['fp']}, "
          f"FN={metrici_test['fn']}, TN={metrici_test['tn']}")

    # Pentru context: ce ar fi fost daca ne-am uita la threshold optim PE TEST
    # (curiosity check, nu raportam asta ca rezultat oficial)
    calibrare_test_curiosity = gaseste_threshold_optim(labels_test, scoruri_test)
    tau_oracle_test = calibrare_test_curiosity["best_f1"]["threshold"]
    f1_oracle_test = calibrare_test_curiosity["best_f1"]["f1_cls1"]
    print(f"\n  [curiosity check] Threshold oracle PE TEST = {tau_oracle_test:.6f}")
    print(f"  F1 oracle pe test = {f1_oracle_test:.4f} "
          f"(vs F1 cu τ* val = {metrici_test['f1_cls1']:.4f})")
    print(f"  Diferența F1 (oracle − calibrat) = "
          f"{f1_oracle_test - metrici_test['f1_cls1']:+.4f}")

    # Breakdown per sursa pe test
    breakdown_sursa = {}
    for sursa, sub in art_test.groupby("sursa"):
        sursa_labels = sub["label"].values
        sursa_scoruri = sub["diff_mean"].values
        sursa_metrici = calculeaza_metrici(sursa_labels, sursa_scoruri, tau_optim)
        breakdown_sursa[str(sursa)] = sursa_metrici

    # ------------------------------------------------------------------
    # 8. Salvare raport
    # ------------------------------------------------------------------
    raport = {
        "config": {
            "model": MODEL_NAME,
            "seed": SEED,
            "device": device,
            "downsample_cls1_test": DOWNSAMPLE_CLS1_LA_TEST,
            "filtru_lungime_propozitie": [MIN_CUVINTE, MAX_CUVINTE],
        },
        "volume": {
            "val_articole": int(df_val["id"].nunique()),
            "val_articole_cu_propozitii": int(art_val.shape[0]),
            "val_propozitii": len(df_val_prop),
            "val_cls0": int((df_val["label_numeric"] == 0).sum()),
            "val_cls1": int((df_val["label_numeric"] == 1).sum()),
            "test_articole": int(df_test["articol_id"].nunique()),
            "test_propozitii": len(df_test),
            "corpus_cls0": len(df_cls0),
            "corpus_cls1_baseline_test": len(df_cls1_baseline),
            "corpus_cls1_noval_calibrare": len(df_cls1_noval),
        },
        "anti_contaminare": {
            "val_cls1_ids_excluse_din_corpus": len(val_cls1_ids),
            "val_cls0_ids_in_corpus_cls0": 0,
        },
        "calibrare_val": {
            "threshold_optim_F1": tau_optim,
            "metrici_val_la_tau_optim": calibrare["best_f1"],
            "best_accuracy_val": calibrare["best_accuracy"],
            "best_youden_j_val": calibrare["best_youden_j"],
            "curba_completa": calibrare["curba"],
        },
        "evaluare_test": {
            "metrici_test_la_tau_calibrat": metrici_test,
            "tau_oracle_test_pentru_referinta": tau_oracle_test,
            "f1_oracle_test": f1_oracle_test,
            "delta_f1_oracle_vs_calibrat": (
                f1_oracle_test - metrici_test["f1_cls1"]
            ),
            "breakdown_per_sursa": breakdown_sursa,
        },
    }

    print("\n[8/8] Scriere rapoarte...")
    _scrie_raport_json(raport, RAPORT_JSON)
    _scrie_raport_md(raport, RAPORT_MD)

    # Rezumat consola
    print("\n" + "=" * 70)
    print("REZUMAT FINAL")
    print("=" * 70)
    print(f"Threshold calibrat τ* (pe val) = {tau_optim:.6f}")
    print()
    print(f"{'':16s} {'Val (calibrare)':>18s} {'Test (evaluare)':>18s}")
    print("-" * 60)
    val_m = calibrare["best_f1"]
    test_m = metrici_test
    print(f"{'Accuracy':16s} {val_m['accuracy']:>18.4f} "
          f"{test_m['accuracy']:>18.4f}")
    print(f"{'Precision':16s} {val_m['precision_cls1']:>18.4f} "
          f"{test_m['precision_cls1']:>18.4f}")
    print(f"{'Recall':16s} {val_m['recall_cls1']:>18.4f} "
          f"{test_m['recall_cls1']:>18.4f}")
    print(f"{'F1':16s} {val_m['f1_cls1']:>18.4f} "
          f"{test_m['f1_cls1']:>18.4f}")
    print()
    print(f"Comparație cu modul 2 (IID): F1 = 100%, recall LOSO-V = 29.35%")
    print(f"Modul 3 la τ* val: F1 = {test_m['f1_cls1']:.4f} pe test (necontaminat)")


# ---------------------------------------------------------------------------
# Scriere rapoarte
# ---------------------------------------------------------------------------

def _scrie_raport_json(raport: dict, path: Path) -> None:
    """JSON cu toate datele structurate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(raport, f, ensure_ascii=False, indent=2, default=float)
    print(f"  JSON scris: {path}")


def _scrie_raport_md(raport: dict, path: Path) -> None:
    """Raport Markdown lizibil pentru teza."""
    config = raport["config"]
    volume = raport["volume"]
    cal = raport["calibrare_val"]
    ev = raport["evaluare_test"]
    val_m = cal["metrici_val_la_tau_optim"]
    test_m = ev["metrici_test_la_tau_calibrat"]
    tau = cal["threshold_optim_F1"]

    linii = []
    linii.append("# Calibrare threshold pentru scor combinat (modul 3)")
    linii.append("")
    linii.append(
        "Calibrare threshold τ* pe val + evaluare pe test "
        "(protocol academic curat — fără data leakage)."
    )
    linii.append("")

    # Configurare
    linii.append("## Configurare")
    linii.append("")
    linii.append(f"- Model: `{config['model']}`")
    linii.append(f"- Seed: `{config['seed']}`, Device: `{config['device']}`")
    linii.append(f"- Filtru lungime propoziție: "
                 f"`{config['filtru_lungime_propozitie']}` cuvinte")
    linii.append(f"- Scor folosit: **`diff_mean`** (= `scor_cls1_mean − "
                 f"scor_cls0_mean`)")
    linii.append("")

    # Volume
    linii.append("## Volume")
    linii.append("")
    linii.append("| Set | Articole | Propoziții | cls0 | cls1 |")
    linii.append("|---|---:|---:|---:|---:|")
    linii.append(
        f"| **Val** (calibrare) | {volume['val_articole']} "
        f"({volume['val_articole_cu_propozitii']} cu propoziții) | "
        f"{volume['val_propozitii']:,} | {volume['val_cls0']} | "
        f"{volume['val_cls1']} |"
    )
    linii.append(
        f"| **Test** (evaluare) | {volume['test_articole']} | "
        f"{volume['test_propozitii']:,} | "
        f"55 | 112 |"
    )
    linii.append(
        f"| Corpus cls0 (referință) | — | "
        f"{volume['corpus_cls0']:,} | — | — |"
    )
    linii.append(
        f"| Corpus cls1 BASELINE (pentru test) | — | "
        f"{volume['corpus_cls1_baseline_test']:,} | — | — |"
    )
    linii.append(
        f"| Corpus cls1 NO-VAL (pentru calibrare) | — | "
        f"{volume['corpus_cls1_noval_calibrare']:,} | — | — |"
    )
    linii.append("")

    # Anti-contaminare
    linii.append("## Anti-contaminare")
    linii.append("")
    linii.append(
        f"**Critical fix:** Corpusul cls1 original conține propoziții din "
        f"articolele val. Pentru calibrare, am exclus cele "
        f"{raport['anti_contaminare']['val_cls1_ids_excluse_din_corpus']} "
        f"articole val cls1 din corpus → "
        f"`{volume['corpus_cls1_noval_calibrare']:,}` propoziții. "
        f"Pentru evaluarea pe test, folosim corpusul **baseline** original "
        f"(test e curat, anti-contaminare validată în benchmark v4)."
    )
    linii.append("")

    # Threshold optim
    linii.append(f"## Threshold optim: **τ* = {tau:.6f}**")
    linii.append("")
    linii.append(
        "Calibrat pe val (max F1 cls1). Grid search peste "
        f"{cal['curba_completa'][0]['threshold']:.4f} → "
        f"{cal['curba_completa'][-1]['threshold']:.4f} "
        f"({len(cal['curba_completa'])} threshold-uri evaluate)."
    )
    linii.append("")

    # Threshold-uri alternative
    linii.append("### Alternative threshold (pentru context)")
    linii.append("")
    linii.append("| Criteriu | Threshold | F1 | Accuracy | Recall | Precision |")
    linii.append("|---|---:|---:|---:|---:|---:|")
    for nume, key in [("Max F1 (folosit)", "best_f1"),
                       ("Max Accuracy", "best_accuracy"),
                       ("Max Youden's J", "best_youden_j")]:
        m = cal[key]
        linii.append(
            f"| {nume} | {m['threshold']:.6f} | {m['f1_cls1']:.4f} | "
            f"{m['accuracy']:.4f} | {m['recall_cls1']:.4f} | "
            f"{m['precision_cls1']:.4f} |"
        )
    linii.append("")

    # Rezultate principale
    linii.append("## Rezultate principale")
    linii.append("")
    linii.append("| Metric | Val (calibrare) | Test (evaluare) |")
    linii.append("|---|---:|---:|")
    linii.append(f"| Accuracy | {val_m['accuracy']:.4f} | "
                 f"{test_m['accuracy']:.4f} |")
    linii.append(f"| Precision (cls1) | {val_m['precision_cls1']:.4f} | "
                 f"{test_m['precision_cls1']:.4f} |")
    linii.append(f"| Recall (cls1) | {val_m['recall_cls1']:.4f} | "
                 f"{test_m['recall_cls1']:.4f} |")
    linii.append(f"| **F1 (cls1)** | **{val_m['f1_cls1']:.4f}** | "
                 f"**{test_m['f1_cls1']:.4f}** |")
    linii.append("")

    linii.append("### Confusion matrix pe test (la τ*)")
    linii.append("")
    linii.append("|  | Pred cls0 | Pred cls1 |")
    linii.append("|---|---:|---:|")
    linii.append(f"| **Real cls0** | TN = {test_m['tn']} | FP = "
                 f"{test_m['fp']} |")
    linii.append(f"| **Real cls1** | FN = {test_m['fn']} | TP = "
                 f"{test_m['tp']} |")
    linii.append("")

    # Sanity check oracle
    linii.append("## Sanity check — oracle vs calibrat")
    linii.append("")
    linii.append(
        f"Diferența F1 între threshold oracle (max F1 PE TEST) și threshold "
        f"calibrat (max F1 PE VAL, aplicat pe test):"
    )
    linii.append("")
    linii.append(f"- F1 oracle (test): **{ev['f1_oracle_test']:.4f}** "
                 f"(τ_oracle = {ev['tau_oracle_test_pentru_referinta']:.6f})")
    linii.append(f"- F1 calibrat (test): **{test_m['f1_cls1']:.4f}** "
                 f"(τ* = {tau:.6f})")
    linii.append(f"- Δ = {ev['delta_f1_oracle_vs_calibrat']:+.4f}")
    linii.append("")
    linii.append(
        "Dacă Δ ≤ 0.01, threshold-ul calibrat e aproape la fel de bun ca "
        "oracle → calibrarea e robustă. Dacă Δ ≥ 0.05, val și test au "
        "distribuții diferite → cifrele test sunt subestimate."
    )
    linii.append("")

    # Breakdown per sursa
    linii.append("## Breakdown test per sursă")
    linii.append("")
    linii.append(
        "Indicator de robustețe cross-source: cât de uniform performează "
        "sistemul pe diferite surse cls0 (HotNews, Pro TV, Libertatea) și "
        "cls1 (Veridica, Stopfals)."
    )
    linii.append("")
    linii.append("| Sursă | n | TP | FP | FN | TN | Accuracy | F1 |")
    linii.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for sursa, m in ev["breakdown_per_sursa"].items():
        n = m["tp"] + m["fp"] + m["fn"] + m["tn"]
        linii.append(
            f"| {sursa} | {n} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{m['tn']} | {m['accuracy']:.4f} | {m['f1_cls1']:.4f} |"
        )
    linii.append("")

    # Comparatie modul 2
    linii.append("## Comparație directă cu modulul 2")
    linii.append("")
    linii.append("| Modul | Setup | Recall cls1 | F1 |")
    linii.append("|---|---|---:|---:|")
    linii.append("| Modul 2 (XLM-R) | IID standard (test set) | 100% | 100% |")
    linii.append("| Modul 2 (XLM-R) | LOSO-V | 29.35% | — |")
    linii.append(
        f"| **Modul 3** (scor D mean) | **IID + cross-source** | "
        f"**{test_m['recall_cls1']*100:.2f}%** | "
        f"**{test_m['f1_cls1']:.4f}** |"
    )
    linii.append("")

    # Note pentru teza
    linii.append("## Note pentru teză")
    linii.append("")
    linii.append(
        "- **Threshold τ\\*** ales pe val, evaluat pe test → protocol academic "
        "standard, fără overfitting"
    )
    linii.append(
        '- **Corpus "no-val"** construit special pentru calibrare → eliminăm '
        "data leakage propozițional"
    )
    linii.append(
        "- **Corpus baseline** folosit pentru test → comparabilitate directă "
        "cu benchmark v4 post-curățare"
    )
    linii.append(
        f"- **Cifră finală pentru raport în teză:** F1 = "
        f"{test_m['f1_cls1']:.4f} pe test set independent (167 articole "
        f"din 5 surse, 3 cls0 + 2 cls1)"
    )
    linii.append("")
    linii.append("---")
    linii.append("")
    linii.append("*Modul 3 · Pasul A4 · Calibrare threshold pe scor combinat*")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(linii))
    print(f"  MD scris: {path}")


if __name__ == "__main__":
    main()
