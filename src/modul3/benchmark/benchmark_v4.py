"""
Benchmark v4 — scor granular per propozitie vs corpus propagandistic.

Ipoteza testata:
----------------
Propozitii dintr-un articol propagandist (cls1) vor avea similaritate
cosine MAI MARE cu propozitii din corpusul propagandistic (Veridica+Stopfals)
decat articole credibile (cls0).

Doua teste rulate in paralel:
  Test A: AUC pe `scor_cls1` izolat (ipoteza: cls1 > cls0)
  Test D: AUC pe `scor_cls1 - scor_cls0` (diferenta combinata)

Metodologie:
------------
1. Embed ambele corpusuri cu mpnet (castigator din v3)
   - cls0: 5,290 prop. (corpus credibil Digi24+G4Media)
   - cls1: 6,048 prop. → downsample la 5,290 pentru paritate
2. Pentru fiecare propozitie din test set:
   - cosine max vs corpus cls0 → scor_cls0
   - cosine max vs corpus cls1 → scor_cls1
3. Agregare la articol: mean, min, p10
4. Calcul AUC-ROC per agregare per test
5. Comparatie cu v3 (unde AUC pe cls0-only a fost 0.552)

Fix-uri metodologice din feedback v1→v3:
- Downsample cls1 la 5,290 (paritate exacta cu cls0, seed=42)
- Validare anti-contaminare INAINTE de embedding (articole test ≠ corpus)
- Cache embeddings pe disc (invalidat automat la schimbare continut)
- Gap Veridica-Stopfals raportat pentru detectia de stylistic leakage

Output:
  - findings/benchmark_v4.md + .json (rezultate + interpretare)
  - data/processed/embeddings_cache/*.npy (embeddings cache-uite)

Utilizare:
  python scripts/benchmark_v4.py
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurare globala
# ---------------------------------------------------------------------------

# Cai fisiere
CORPUS_CLS0_PATH = Path("data/processed/propozitii_cls0_corpus.parquet")
CORPUS_CLS1_PATH = Path("data/processed/propozitii_cls1_corpus_v2.parquet")
TEST_SET_PATH = Path("data/processed/subset_benchmark_v3.parquet")

CACHE_DIR = Path("data/processed/embeddings_cache")
RAPORT_MD = Path("findings/benchmark_v4.md")
RAPORT_JSON = Path("findings/benchmark_v4.json")

# Parametri benchmark (conform HANDOFF_modul3_continuare.md)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SEED = 42
BATCH_SIZE = 32
DOWNSAMPLE_CLS1_LA = 5_290  # paritate cu corpus cls0

# Agregari de raportat (aplicate pe toate propozitiile unui articol)
AGREGARI_ARTICOL = ["mean", "min", "p10"]


# ---------------------------------------------------------------------------
# Setup reproducibilitate
# ---------------------------------------------------------------------------

def seteaza_seed(seed: int = SEED) -> None:
    """Fixeaza seed-ul pentru numpy, random, torch (daca e disponibil)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def selecteaza_device() -> str:
    """Alege MPS (Apple Silicon) / CUDA / CPU in ordinea asta de preferinta."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Cache embeddings
# ---------------------------------------------------------------------------

def calculeaza_hash_corpus(texts: Iterable[str], model_name: str) -> str:
    """
    Construieste o cheie determinista pentru cache: hash pe continutul textelor
    + numele modelului. Daca se schimba vreun text sau modelul, cache-ul
    se invalideaza automat.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\n")
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def incarca_sau_calculeaza_embeddings(
    texts: list[str],
    nume_corpus: str,
    model,
    device: str,
) -> np.ndarray:
    """
    Incarca embeddings din cache daca exista, altfel calculeaza si salveaza.

    Cheia cache include hash-ul continutului corpus + model name, deci daca
    utilizatorul schimba corpusul cls1 de la v2 la v3, cache-ul se invalideaza
    automat si embeddings-urile sunt recalculate.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hash_corpus = calculeaza_hash_corpus(texts, MODEL_NAME)
    cache_path = CACHE_DIR / f"{nume_corpus}_{hash_corpus}.npy"

    if cache_path.exists():
        print(f"  [cache HIT] {cache_path.name}")
        embeddings = np.load(cache_path)
        # Sanity check: numarul de embeddings trebuie sa corespunda textelor
        assert embeddings.shape[0] == len(texts), (
            f"Cache corupt: {embeddings.shape[0]} embeddings vs "
            f"{len(texts)} texte. Șterge {cache_path} și re-rulează."
        )
        return embeddings

    print(f"  [cache MISS] calculez pe {device}...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalizare L2 → cosine = dot product
        device=device,
    )
    elapsed = time.time() - t0
    print(f"  [done] {len(texts):,} prop. în {elapsed:.1f}s "
          f"({len(texts) / elapsed:.0f} prop./s)")

    # Persista pe disc pentru rularile viitoare
    np.save(cache_path, embeddings)
    print(f"  [cache SAVE] {cache_path.name}")
    return embeddings


# ---------------------------------------------------------------------------
# Validare anti-contaminare
# ---------------------------------------------------------------------------

def valideaza_anti_contaminare(
    df_test: pd.DataFrame,
    df_cls0: pd.DataFrame,
    df_cls1: pd.DataFrame,
) -> dict:
    """
    Verifica ca nicio propozitie din test set nu apare si in corpusuri.

    Raporteaza orice suprapunere. Benchmark-ul v1 a fost invalidat de
    exact aceasta problema (articole test ∈ corpus), deci verificarea e
    obligatorie inainte de embedding.
    """
    print("\n[anti-contaminare] Verific suprapuneri articol_id...")

    test_ids = set(df_test["articol_id"].unique())
    cls0_ids = set(df_cls0["articol_id"].unique())
    cls1_ids = set(df_cls1["articol_id"].unique())

    overlap_cls0 = test_ids & cls0_ids
    overlap_cls1 = test_ids & cls1_ids

    raport = {
        "test_articole_unice": len(test_ids),
        "cls0_articole_unice": len(cls0_ids),
        "cls1_articole_unice": len(cls1_ids),
        "suprapunere_test_cls0": len(overlap_cls0),
        "suprapunere_test_cls1": len(overlap_cls1),
        "exemple_overlap_cls0": sorted(overlap_cls0)[:10],
        "exemple_overlap_cls1": sorted(overlap_cls1)[:10],
    }

    if overlap_cls0 or overlap_cls1:
        print(f"  ⚠️  CONTAMINARE DETECTATĂ:")
        if overlap_cls0:
            print(f"      test ∩ cls0 = {len(overlap_cls0)} articole: "
                  f"{sorted(overlap_cls0)[:5]}...")
        if overlap_cls1:
            print(f"      test ∩ cls1 = {len(overlap_cls1)} articole: "
                  f"{sorted(overlap_cls1)[:5]}...")
        raise RuntimeError(
            "Contaminare detectată între test set și corpusuri. "
            "Benchmark-ul ar da rezultate invalidate. Vezi raport anti-contaminare."
        )

    print(f"  ✓ zero suprapuneri (test={len(test_ids)}, "
          f"cls0={len(cls0_ids)}, cls1={len(cls1_ids)})")
    return raport


# ---------------------------------------------------------------------------
# Scoruri per propozitie (cosine max)
# ---------------------------------------------------------------------------

def scor_cosine_max_batch(
    emb_test: np.ndarray,
    emb_corpus: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Pentru fiecare rand din emb_test, calculeaza similaritate max cu toate
    randurile din emb_corpus. Procesat in batch-uri ca sa evite OOM.

    Presupune embeddings normalizate L2 (cosine = produs scalar).

    Returneaza: vector shape (len(emb_test),) cu scorurile max.
    """
    n_test = emb_test.shape[0]
    scoruri_max = np.zeros(n_test, dtype=np.float32)

    for i in range(0, n_test, batch_size):
        batch = emb_test[i:i + batch_size]  # (b, d)
        # sim = batch @ emb_corpus.T  → shape (b, n_corpus)
        sim = batch @ emb_corpus.T
        # max pe axa corpus → scor pentru fiecare propozitie din batch
        scoruri_max[i:i + batch_size] = sim.max(axis=1)

    return scoruri_max


# ---------------------------------------------------------------------------
# Agregare scoruri la articol
# ---------------------------------------------------------------------------

def agrega_la_articol(
    df_test: pd.DataFrame,
    scoruri_cls0: np.ndarray,
    scoruri_cls1: np.ndarray,
) -> pd.DataFrame:
    """
    Ataseaza scorurile propozitiilor la df_test si agrega la nivel de articol.

    Returneaza DataFrame indexat pe articol_id cu coloanele:
      - label, sursa
      - scor_cls0_mean, scor_cls0_min, scor_cls0_p10
      - scor_cls1_mean, scor_cls1_min, scor_cls1_p10
      - diff_mean, diff_min, diff_p10  (scor_cls1 − scor_cls0, pentru Test D)
    """
    df = df_test.copy()
    df["scor_cls0"] = scoruri_cls0
    df["scor_cls1"] = scoruri_cls1

    # Agregari pentru mean, min, p10
    def p10(series: pd.Series) -> float:
        """Percentila 10 — mai robust la outlieri decat min."""
        return float(np.percentile(series, 10))

    articol_scoruri = df.groupby("articol_id").agg(
        label=("label_numeric", "first"),
        sursa=("sursa_site", "first"),
        nr_prop=("propozitie", "count"),
        scor_cls0_mean=("scor_cls0", "mean"),
        scor_cls0_min=("scor_cls0", "min"),
        scor_cls0_p10=("scor_cls0", p10),
        scor_cls1_mean=("scor_cls1", "mean"),
        scor_cls1_min=("scor_cls1", "min"),
        scor_cls1_p10=("scor_cls1", p10),
    )

    # Test D: diferenta scor_cls1 − scor_cls0 per agregare
    articol_scoruri["diff_mean"] = (
        articol_scoruri["scor_cls1_mean"] - articol_scoruri["scor_cls0_mean"]
    )
    articol_scoruri["diff_min"] = (
        articol_scoruri["scor_cls1_min"] - articol_scoruri["scor_cls0_min"]
    )
    articol_scoruri["diff_p10"] = (
        articol_scoruri["scor_cls1_p10"] - articol_scoruri["scor_cls0_p10"]
    )

    return articol_scoruri


# ---------------------------------------------------------------------------
# Calcul metrici (AUC, gap V-S)
# ---------------------------------------------------------------------------

def calculeaza_auc(labels: np.ndarray, scoruri: np.ndarray) -> float:
    """
    AUC-ROC. Conventie: scor mare = predictie pozitiva (cls1).
    Daca labels contine doar o clasa, returneaza NaN.
    """
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scoruri))


def calculeaza_gap_veridica_stopfals(
    articol_scoruri: pd.DataFrame,
    coloana_scor: str,
) -> dict | None:
    """
    Calculeaza gap-ul intre scorurile medii pe Veridica vs Stopfals (ambele cls1).

    Gap mare (> 0.05) = stylistic leakage — modelul discrimineaza intre cele
    doua surse chiar in aceeasi clasa, semnal rau. Gap mic (~ 0) = bun.

    Returneaza dict cu media per sursa + gap absolut, sau None daca lipseste
    o sursa.
    """
    cls1 = articol_scoruri[articol_scoruri["label"] == 1]
    v = cls1[cls1["sursa"] == "veridica.ro"][coloana_scor]
    s = cls1[cls1["sursa"] == "stopfals.md"][coloana_scor]

    if len(v) == 0 or len(s) == 0:
        return None

    return {
        "veridica_mean": float(v.mean()),
        "stopfals_mean": float(s.mean()),
        "gap": float(v.mean() - s.mean()),
        "n_veridica": int(len(v)),
        "n_stopfals": int(len(s)),
    }


def calculeaza_cohen_d(
    articol_scoruri: pd.DataFrame,
    coloana_scor: str,
) -> float:
    """
    Cohen's d: marimea efectului (cls1 vs cls0). Conventie: pozitiv = cls1 > cls0.
    |d| ≥ 0.8 efect mare, 0.5 mediu, 0.2 mic.
    """
    cls0 = articol_scoruri[articol_scoruri["label"] == 0][coloana_scor]
    cls1 = articol_scoruri[articol_scoruri["label"] == 1][coloana_scor]
    if len(cls0) < 2 or len(cls1) < 2:
        return float("nan")
    # Pooled std (varianta combinata)
    var_pooled = ((len(cls0) - 1) * cls0.var() + (len(cls1) - 1) * cls1.var())
    var_pooled /= (len(cls0) + len(cls1) - 2)
    if var_pooled <= 0:
        return float("nan")
    return float((cls1.mean() - cls0.mean()) / np.sqrt(var_pooled))


# ---------------------------------------------------------------------------
# Rulare benchmark + raport
# ---------------------------------------------------------------------------

def main() -> None:
    """Punct de intrare. Orchestreaza incarcare → embedding → scoring → raport."""
    print("=" * 70)
    print("BENCHMARK v4 — scor granular vs corpus propagandistic")
    print("=" * 70)
    seteaza_seed(SEED)
    device = selecteaza_device()
    print(f"Device: {device} | Seed: {SEED} | Model: {MODEL_NAME}")

    # ------------------------------------------------------------------
    # 1. Incarcare date
    # ------------------------------------------------------------------
    print("\n[1/5] Încărcare date...")
    for p in (CORPUS_CLS0_PATH, CORPUS_CLS1_PATH, TEST_SET_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Fișier lipsă: {p}")

    df_cls0 = pd.read_parquet(CORPUS_CLS0_PATH)
    df_cls1_full = pd.read_parquet(CORPUS_CLS1_PATH)
    df_test = pd.read_parquet(TEST_SET_PATH)

    print(f"  cls0 corpus: {len(df_cls0):,} prop.")
    print(f"  cls1 corpus: {len(df_cls1_full):,} prop. (pre-downsample)")
    print(f"  test set:    {len(df_test):,} prop. "
          f"({df_test['articol_id'].nunique()} articole)")

    # ------------------------------------------------------------------
    # 2. Downsample cls1 la paritate cu cls0
    # ------------------------------------------------------------------
    print(f"\n[2/5] Downsample cls1 la {DOWNSAMPLE_CLS1_LA:,} prop. "
          f"(seed={SEED})...")
    df_cls1 = df_cls1_full.sample(
        n=DOWNSAMPLE_CLS1_LA,
        random_state=SEED,
    ).reset_index(drop=True)
    print(f"  cls1 corpus (după downsample): {len(df_cls1):,} prop.")
    print(f"  Paritate cls0 vs cls1: {len(df_cls0)} vs {len(df_cls1)}")

    # Breakdown sursa dupa downsample
    print(f"  Distribuție sursă cls1 downsampled:")
    for sursa, n in df_cls1["sursa_site"].value_counts().items():
        pct = n / len(df_cls1) * 100
        print(f"    {sursa}: {n:,} ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 3. Validare anti-contaminare (inainte de embedding!)
    # ------------------------------------------------------------------
    raport_contaminare = valideaza_anti_contaminare(df_test, df_cls0, df_cls1)

    # ------------------------------------------------------------------
    # 4. Incarcare model + calcul embeddings
    # ------------------------------------------------------------------
    print("\n[3/5] Încărcare model sentence-transformers...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Instalează sentence-transformers: "
            "pip install sentence-transformers --break-system-packages"
        )

    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  Model încărcat. Dim embedding: {model.get_sentence_embedding_dimension()}")

    print("\n[4/5] Calcul embeddings...")
    print(" → corpus cls0")
    emb_cls0 = incarca_sau_calculeaza_embeddings(
        df_cls0["propozitie"].tolist(),
        nume_corpus="cls0_corpus",
        model=model,
        device=device,
    )
    print(" → corpus cls1 (downsampled)")
    emb_cls1 = incarca_sau_calculeaza_embeddings(
        df_cls1["propozitie"].tolist(),
        nume_corpus="cls1_corpus_v2_downsampled",
        model=model,
        device=device,
    )
    print(" → test set (fără cache; e mic)")
    emb_test = model.encode(
        df_test["propozitie"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    print(f"  Shape-uri: cls0={emb_cls0.shape}, cls1={emb_cls1.shape}, "
          f"test={emb_test.shape}")

    # ------------------------------------------------------------------
    # 5. Scoring + agregare + metrici
    # ------------------------------------------------------------------
    print("\n[5/5] Scoring cosine max + agregare per articol...")
    scoruri_cls0 = scor_cosine_max_batch(emb_test, emb_cls0)
    scoruri_cls1 = scor_cosine_max_batch(emb_test, emb_cls1)
    print(f"  Scoruri cls0: μ={scoruri_cls0.mean():.4f}, "
          f"σ={scoruri_cls0.std():.4f}")
    print(f"  Scoruri cls1: μ={scoruri_cls1.mean():.4f}, "
          f"σ={scoruri_cls1.std():.4f}")

    articol_scoruri = agrega_la_articol(df_test, scoruri_cls0, scoruri_cls1)
    labels = articol_scoruri["label"].values

    # Calculeaza toate metricile relevante intr-o structura unitara
    rezultate = {
        "test_A_cls1_only": {},   # AUC pe scor_cls1 izolat
        "test_B_cls0_only": {},   # AUC pe scor_cls0 (reproducere v3 pentru comparatie)
        "test_D_diff": {},        # AUC pe scor_cls1 − scor_cls0
    }

    for agr in AGREGARI_ARTICOL:
        col_cls1 = f"scor_cls1_{agr}"
        col_cls0 = f"scor_cls0_{agr}"
        col_diff = f"diff_{agr}"

        # Test A: AUC pe cls1 izolat (conventie: scor mare → cls1)
        rezultate["test_A_cls1_only"][agr] = {
            "auc": calculeaza_auc(labels, articol_scoruri[col_cls1].values),
            "cohen_d": calculeaza_cohen_d(articol_scoruri, col_cls1),
            "gap_veridica_stopfals": calculeaza_gap_veridica_stopfals(
                articol_scoruri, col_cls1
            ),
            "media_cls0": float(articol_scoruri[articol_scoruri["label"] == 0][col_cls1].mean()),
            "media_cls1": float(articol_scoruri[articol_scoruri["label"] == 1][col_cls1].mean()),
        }

        # Test B: AUC pe cls0 izolat — pentru comparatie cu v3
        # In v3 conventia a fost inversata: cls0 e corpus credibil, deci
        # articolele credibile ar trebui sa aiba cos(cls0) mai mare.
        # Aici pastram conventia "scor mare → cls1", deci AUC-ul pe cls0 ar
        # trebui sa fie APROAPE DE 0.5 sau SUB 0.5 (daca are semnal invers).
        rezultate["test_B_cls0_only"][agr] = {
            "auc": calculeaza_auc(labels, articol_scoruri[col_cls0].values),
            "cohen_d": calculeaza_cohen_d(articol_scoruri, col_cls0),
            "gap_veridica_stopfals": calculeaza_gap_veridica_stopfals(
                articol_scoruri, col_cls0
            ),
            "media_cls0": float(articol_scoruri[articol_scoruri["label"] == 0][col_cls0].mean()),
            "media_cls1": float(articol_scoruri[articol_scoruri["label"] == 1][col_cls0].mean()),
        }

        # Test D: AUC pe diferenta
        rezultate["test_D_diff"][agr] = {
            "auc": calculeaza_auc(labels, articol_scoruri[col_diff].values),
            "cohen_d": calculeaza_cohen_d(articol_scoruri, col_diff),
            "gap_veridica_stopfals": calculeaza_gap_veridica_stopfals(
                articol_scoruri, col_diff
            ),
            "media_cls0": float(articol_scoruri[articol_scoruri["label"] == 0][col_diff].mean()),
            "media_cls1": float(articol_scoruri[articol_scoruri["label"] == 1][col_diff].mean()),
        }

    # ------------------------------------------------------------------
    # Scriere rapoarte
    # ------------------------------------------------------------------
    raport_complet = {
        "config": {
            "model": MODEL_NAME,
            "seed": SEED,
            "device": device,
            "downsample_cls1_la": DOWNSAMPLE_CLS1_LA,
        },
        "volume": {
            "corpus_cls0": len(df_cls0),
            "corpus_cls1_full": len(df_cls1_full),
            "corpus_cls1_downsampled": len(df_cls1),
            "test_articole": int(df_test["articol_id"].nunique()),
            "test_propozitii": len(df_test),
            "test_cls0_articole": int(
                articol_scoruri[articol_scoruri["label"] == 0].shape[0]
            ),
            "test_cls1_articole": int(
                articol_scoruri[articol_scoruri["label"] == 1].shape[0]
            ),
        },
        "anti_contaminare": raport_contaminare,
        "rezultate": rezultate,
    }

    _scrie_raport_json(raport_complet, RAPORT_JSON)
    _scrie_raport_md(raport_complet, RAPORT_MD)

    # Rezumat consola
    print("\n" + "=" * 70)
    print("REZUMAT BENCHMARK v4")
    print("=" * 70)
    _printeaza_rezumat(rezultate)
    _printeaza_decizie(rezultate)


def _printeaza_rezumat(rezultate: dict) -> None:
    """Tabel consola cu AUC-urile pentru fiecare test × agregare."""
    print(f"\n{'Test':<25} {'Agregare':<10} {'AUC':>8} {'Cohen d':>10}  Gap V-S")
    print("-" * 70)
    for test_nume, rez_test in rezultate.items():
        for agr, m in rez_test.items():
            gap = m["gap_veridica_stopfals"]
            gap_str = f"{gap['gap']:+.4f}" if gap else "N/A"
            print(f"{test_nume:<25} {agr:<10} "
                  f"{m['auc']:>8.4f} {m['cohen_d']:>+10.4f}  {gap_str}")


def _printeaza_decizie(rezultate: dict) -> None:
    """Aplicabilitate decizie Optiunea A / D / C conform HANDOFF."""
    # Cel mai bun AUC peste toate agregarile pentru Test A
    auc_max_A = max(m["auc"] for m in rezultate["test_A_cls1_only"].values())
    auc_max_D = max(m["auc"] for m in rezultate["test_D_diff"].values())

    print(f"\n  Best Test A (cls1 izolat): AUC = {auc_max_A:.4f}")
    print(f"  Best Test D (diferență):   AUC = {auc_max_D:.4f}")

    if auc_max_A >= 0.75:
        dec = "✓ Opțiunea A VALIDATĂ — continuăm cu D pentru rigurozitate finală"
    elif auc_max_A < 0.65:
        dec = ("✗ Opțiunea A NU e viabilă — trecem la Opțiunea C "
               "(instrument asistiv vizual)")
    else:
        dec = (f"⚠ Opțiunea A în zona gri (0.65 ≤ AUC < 0.75) — "
               f"discuție trade-off timp vs rigurozitate")

    print(f"\n  Decizie: {dec}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Formatare rapoarte
# ---------------------------------------------------------------------------

def _scrie_raport_json(raport: dict, path: Path) -> None:
    """Scrie JSON-ul cu toate rezultatele (pentru comparatii programatice)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(raport, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n  JSON scris: {path}")


def _scrie_raport_md(raport: dict, path: Path) -> None:
    """Scrie raport Markdown lizibil cu tabele si interpretare."""
    config = raport["config"]
    volume = raport["volume"]
    rezultate = raport["rezultate"]

    linii: list[str] = []
    linii.append("# Benchmark v4 — scor granular vs corpus propagandistic")
    linii.append("")
    linii.append("## Configurare")
    linii.append("")
    linii.append(f"- Model: `{config['model']}`")
    linii.append(f"- Device: `{config['device']}`")
    linii.append(f"- Seed: `{config['seed']}`")
    linii.append(f"- Downsample cls1: `{config['downsample_cls1_la']:,}` "
                 f"(paritate cu cls0)")
    linii.append("")

    linii.append("## Volume")
    linii.append("")
    linii.append(f"- Corpus cls0: **{volume['corpus_cls0']:,}** prop. (congelat)")
    linii.append(f"- Corpus cls1 full: {volume['corpus_cls1_full']:,} prop. "
                 f"→ downsampled la **{volume['corpus_cls1_downsampled']:,}** "
                 f"prop. (seed={config['seed']})")
    linii.append(f"- Test set: **{volume['test_articole']}** articole "
                 f"({volume['test_propozitii']:,} prop.) — "
                 f"{volume['test_cls0_articole']} cls0 + "
                 f"{volume['test_cls1_articole']} cls1")
    linii.append("")

    linii.append("## Validare anti-contaminare")
    linii.append("")
    ac = raport["anti_contaminare"]
    linii.append(f"- Suprapunere test ∩ cls0: **{ac['suprapunere_test_cls0']}** "
                 f"articole")
    linii.append(f"- Suprapunere test ∩ cls1: **{ac['suprapunere_test_cls1']}** "
                 f"articole")
    if ac["suprapunere_test_cls0"] == 0 and ac["suprapunere_test_cls1"] == 0:
        linii.append("- ✓ Zero contaminare — benchmark valid")
    linii.append("")

    # Tabele per test
    linii.append("## Rezultate")
    linii.append("")
    linii.append("**Convenție:** scor mare = predicție cls1 (propagandist). "
                 "AUC > 0.5 = semnal util; 0.5 = aleator; < 0.5 = semnal inversat.")
    linii.append("")

    descrieri = {
        "test_A_cls1_only": (
            "**Test A — scor_cls1 izolat** (ipoteza principală Opțiunea A): "
            "articolele propagandiste au similaritate mai mare cu corpusul "
            "propagandistic decât cele credibile?"
        ),
        "test_B_cls0_only": (
            "**Test B — scor_cls0 izolat** (reproducere v3 pentru comparație): "
            "rezultatul din v3 a fost AUC = 0.552 pe această configurație. "
            "Replicare cu corpusul extins vs v3 pentru sanity-check."
        ),
        "test_D_diff": (
            "**Test D — diferență cls1 − cls0** (Opțiunea D combinată): "
            "scor compus care folosește ambele corpusuri. Dacă "
            "`scor_cls1 > scor_cls0`, articolul e mai aproape de propagandă "
            "decât de presă credibilă."
        ),
    }

    for test_key, descriere in descrieri.items():
        linii.append(f"### {descriere}")
        linii.append("")
        linii.append("| Agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | "
                     "Gap V−S |")
        linii.append("|---|---:|---:|---:|---:|---:|")
        for agr, m in rezultate[test_key].items():
            gap = m["gap_veridica_stopfals"]
            gap_str = f"{gap['gap']:+.4f}" if gap else "N/A"
            linii.append(
                f"| {agr} | {m['auc']:.4f} | {m['cohen_d']:+.4f} | "
                f"{m['media_cls0']:.4f} | {m['media_cls1']:.4f} | "
                f"{gap_str} |"
            )
        linii.append("")

    # Interpretare automata
    linii.append("## Interpretare automată")
    linii.append("")
    auc_max_A = max(m["auc"] for m in rezultate["test_A_cls1_only"].values())
    auc_max_D = max(m["auc"] for m in rezultate["test_D_diff"].values())
    best_agr_A = max(rezultate["test_A_cls1_only"].items(),
                     key=lambda x: x[1]["auc"])[0]
    best_agr_D = max(rezultate["test_D_diff"].items(),
                     key=lambda x: x[1]["auc"])[0]

    linii.append(f"- **Best Test A:** AUC = {auc_max_A:.4f} "
                 f"(agregare `{best_agr_A}`)")
    linii.append(f"- **Best Test D:** AUC = {auc_max_D:.4f} "
                 f"(agregare `{best_agr_D}`)")
    linii.append("")

    if auc_max_A >= 0.75:
        linii.append("### ✓ Decizie: Opțiunea A validată")
        linii.append("")
        linii.append(
            "Scorul pe corpus propagandistic separă semnificativ articolele "
            "propagandiste de cele credibile. Continuăm cu Test D pentru "
            "rigurozitate finală și integrare în scor combinat (modulul 5)."
        )
    elif auc_max_A < 0.65:
        linii.append("### ✗ Decizie: Opțiunea A nu e viabilă")
        linii.append("")
        linii.append(
            "Scorul granular NU separă articolele propagandiste cross-source, "
            "nici cu corpus propagandistic, nici cu credibil. Trecem la "
            "Opțiunea C (tool asistiv vizual): în modulul 5, afișăm per "
            "propoziție top-3 propoziții similare din cls0 și cls1, lăsând "
            "jurnalistul/fact-checker-ul să decidă contextul. "
            "**Finding metodologic 3 (pentru teză):** prima demonstrație "
            "empirică pe română că similaritatea semantică nu separă "
            "dezinformarea nici cu corpus credibil, nici cu corpus "
            "propagandistic — limitare fundamentală a paradigmei "
            "bag-of-sentences."
        )
    else:
        linii.append("### ⚠ Decizie: zona gri")
        linii.append("")
        linii.append(
            f"AUC = {auc_max_A:.4f} e între 0.65 și 0.75. Semnalul există "
            f"dar e slab. Trade-off timp vs rigurozitate: investigăm dacă un "
            f"top-k mean (k=5) îmbunătățește separabilitatea (secundar), sau "
            f"acceptăm ca scor complementar în sistemul combinat cu "
            f"modulul 2 (primar)."
        )

    linii.append("")
    linii.append("## Comparație cu v3")
    linii.append("")
    linii.append("| Metric | v3 (cls0-only) | v4 best (cls1-only) | v4 best (diff) |")
    linii.append("|---|---:|---:|---:|")
    linii.append(f"| AUC | 0.552 (minilm/max/mean) | {auc_max_A:.4f} "
                 f"(mpnet/max/{best_agr_A}) | {auc_max_D:.4f} "
                 f"(mpnet/max/{best_agr_D}) |")
    linii.append("")
    linii.append(
        "Notă: v3 a raportat 0.552 cu minilm (model mai mic). v4 folosește "
        "mpnet (model mai mare, ales câștigător în v3)."
    )
    linii.append("")
    linii.append("---")
    linii.append("")
    linii.append("*Modul 3 · Pasul A2 · Benchmark v4*")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(linii))
    print(f"  MD scris: {path}")


if __name__ == "__main__":
    main()
