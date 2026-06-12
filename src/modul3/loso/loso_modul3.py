"""
LOSO pe modulul 3 — testul cross-source ultim pentru scorul granular.

Context:
--------
Modulul 2 (clasificator XLM-R) a expus stylistic fingerprint leakage:
recall cls1 LOSO-V = 29.35% (drop 70pp fata de IID standard 100%).
Decizia metodologica: daca modulul 3 (scor granular cosine) rezista
cross-source, compenseaza limitarea modulului 2.

Acest script testeaza exact asta. Doua scenarii:

  LOSO-V (Leave-One-Source-Out cu Veridica scoasa):
    Corpus cls1 = doar Stopfals (~450 prop.)
    Test set:    cls0 (HotNews+Pro TV+Libertatea) + cls1 (99 Veridica + 13 Stopfals)
    Intrebare: daca modelul nu vede Veridica in corpus, recunoaste articolele
              Veridica din test prin similaritate cu Stopfals?

  LOSO-S (Leave-One-Source-Out cu Stopfals scos):
    Corpus cls1 = doar Veridica (~5,632 prop.)
    Test set:    identic
    Intrebare: simetric — modelul recunoaste articolele Stopfals prin
              similaritate cu Veridica?

Configuratie:
  - Varianta B (recomandata): cls0 intact (5,290 prop.), nu downsample-am la
    paritate. Reflecta scenariul real (mai multe surse credibile cunoscute).
  - Test set: subset_benchmark_v3_curat.parquet (post-cookie cleanup)
  - Model: mpnet (acelasi ca in benchmark v4)
  - Reutilizam embeddings cache (corpus cls0 + cls1 full) — filtram la indecsi

Output:
  - findings/loso_modul3.md + .json (rezultate, comparatie, decizie)

Utilizare:
  python scripts/loso_modul3.py
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
TEST_SET_PATH = Path("data/processed/subset_benchmark_v3_curat.parquet")
CACHE_DIR = Path("data/processed/embeddings_cache")

RAPORT_MD = Path("findings/loso_modul3.md")
RAPORT_JSON = Path("findings/loso_modul3.json")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SEED = 42
DOWNSAMPLE_CLS1_LA = 5_290  # pentru BASELINE standard (referinta)


# ---------------------------------------------------------------------------
# Utilities (copiate din benchmark_v4 — script standalone)
# ---------------------------------------------------------------------------

def seteaza_seed(seed: int = SEED) -> None:
    """Fixeaza seed-ul pentru reproductibilitate (random + numpy)."""
    random.seed(seed)
    np.random.seed(seed)


def calculeaza_hash_corpus(texts, model_name: str) -> str:
    """Recalculeaza hash identic cu cel din benchmark_v4 → gaseste cache valid."""
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\n")
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def incarca_embeddings_cache(texts: list[str], nume: str) -> np.ndarray:
    """
    Incarca embeddings din cache. Arunca FileNotFoundError daca nu exista.
    Cheia trebuie sa fie identica cu cea din benchmark_v4 (acelasi continut +
    acelasi nume = acelasi hash).
    """
    h = calculeaza_hash_corpus(texts, MODEL_NAME)
    cache_path = CACHE_DIR / f"{nume}_{h}.npy"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache lipsă: {cache_path}. "
            f"Rulează benchmark_v4_post_curatare.py mai întâi."
        )
    emb = np.load(cache_path)
    assert emb.shape[0] == len(texts), (
        f"Cache corupt pe {nume}: {emb.shape[0]} embeddings vs "
        f"{len(texts)} texte"
    )
    return emb


def scor_cosine_max(emb_test: np.ndarray, emb_corpus: np.ndarray,
                    batch: int = 256) -> np.ndarray:
    """Cosine max per propozitie (presupune embeddings normalizate L2)."""
    n = emb_test.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch):
        b = emb_test[i:i + batch]
        out[i:i + batch] = (b @ emb_corpus.T).max(axis=1)
    return out


def auc(labels: np.ndarray, scoruri: np.ndarray) -> float:
    """AUC-ROC standard."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scoruri))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Marimea efectului Cohen's d (b − a) / std_pooled."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_pooled = ((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
    var_pooled /= (len(a) + len(b) - 2)
    if var_pooled <= 0:
        return float("nan")
    return float((b.mean() - a.mean()) / np.sqrt(var_pooled))


def recall_la_threshold(labels: np.ndarray, scoruri: np.ndarray,
                        threshold: float) -> dict:
    """
    Recall pe cls1 + precision la un threshold dat (scor > threshold = cls1).
    Util pentru comparatie directa cu modulul 2 (recall cls1 = 29.35% LOSO-V).
    """
    pred = (scoruri > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    recall_cls1 = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision_cls1 = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {
        "threshold": threshold,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall_cls1": recall_cls1,
        "precision_cls1": precision_cls1,
    }


# ---------------------------------------------------------------------------
# Agregare scoruri la nivel de articol + calcul metrici
# ---------------------------------------------------------------------------

def agrega_si_scoreaza(
    df_test: pd.DataFrame,
    sc_cls0_prop: np.ndarray,
    sc_cls1_prop: np.ndarray,
) -> pd.DataFrame:
    """
    Ataseaza scoruri propozitie la df_test, agrega la articol (mean/min/p10),
    calculeaza diff. Returneaza DataFrame cu un rand per articol.
    """
    df = df_test.copy().reset_index(drop=True)
    df["scor_cls0"] = sc_cls0_prop
    df["scor_cls1"] = sc_cls1_prop

    def p10(s):
        return float(np.percentile(s, 10))

    art = df.groupby("articol_id").agg(
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
    art["diff_mean"] = art["scor_cls1_mean"] - art["scor_cls0_mean"]
    art["diff_min"] = art["scor_cls1_min"] - art["scor_cls0_min"]
    art["diff_p10"] = art["scor_cls1_p10"] - art["scor_cls0_p10"]
    return art


def calcul_metrici_complete(art_scoruri: pd.DataFrame) -> dict:
    """
    Pentru fiecare combinatie (Test A/B/D × agregare mean/min/p10):
    AUC, Cohen's d, μ(cls0), μ(cls1).

    Plus: breakdown per sursa pe cls1 (Veridica vs Stopfals) — semnatura
    pentru detectarea leak-ului intre cele doua surse.
    """
    labels = art_scoruri["label"].values
    rezultate = {}

    teste = {
        "test_A_cls1_only": ["scor_cls1_mean", "scor_cls1_min", "scor_cls1_p10"],
        "test_B_cls0_only": ["scor_cls0_mean", "scor_cls0_min", "scor_cls0_p10"],
        "test_D_diff": ["diff_mean", "diff_min", "diff_p10"],
    }

    for test_nume, coloane in teste.items():
        rezultate[test_nume] = {}
        for col in coloane:
            agr = col.split("_")[-1]  # mean/min/p10
            cls0 = art_scoruri[art_scoruri["label"] == 0][col].values
            cls1 = art_scoruri[art_scoruri["label"] == 1][col].values

            # Breakdown cls1 per sursa (Veridica vs Stopfals)
            cls1_v = art_scoruri[
                (art_scoruri["label"] == 1) &
                (art_scoruri["sursa"] == "veridica.ro")
            ][col].values
            cls1_s = art_scoruri[
                (art_scoruri["label"] == 1) &
                (art_scoruri["sursa"] == "stopfals.md")
            ][col].values

            rezultate[test_nume][agr] = {
                "auc": auc(labels, art_scoruri[col].values),
                "cohen_d": cohen_d(cls0, cls1),
                "media_cls0": float(cls0.mean()) if len(cls0) else float("nan"),
                "media_cls1": float(cls1.mean()) if len(cls1) else float("nan"),
                "media_cls1_veridica": (
                    float(cls1_v.mean()) if len(cls1_v) else float("nan")
                ),
                "media_cls1_stopfals": (
                    float(cls1_s.mean()) if len(cls1_s) else float("nan")
                ),
                "n_veridica": int(len(cls1_v)),
                "n_stopfals": int(len(cls1_s)),
            }

    return rezultate


# ---------------------------------------------------------------------------
# Pipeline LOSO
# ---------------------------------------------------------------------------

def ruleaza_scenariu(
    nume: str,
    descriere: str,
    df_cls1_filtrat: pd.DataFrame,
    emb_cls1_filtrat: np.ndarray,
    df_test: pd.DataFrame,
    emb_test: np.ndarray,
    emb_cls0: np.ndarray,
) -> dict:
    """
    Ruleaza un scenariu LOSO (corpus cls1 deja filtrat).

    Argumente:
      nume: identificator scurt (ex: "loso_v", "loso_s", "baseline_standard")
      descriere: text pentru raport
      df_cls1_filtrat: DataFrame cu propozitii cls1 din corpus (dupa filtrare)
      emb_cls1_filtrat: embeddings corespunzatoare (deja filtrate la indici)
      df_test: test set
      emb_test: embeddings test set
      emb_cls0: embeddings corpus cls0 (NU filtrate, intact)

    Returneaza dict cu metrici complete.
    """
    print(f"\n  → {nume}: {descriere}")
    print(f"    Corpus cls0: {emb_cls0.shape[0]:,} prop.")
    print(f"    Corpus cls1 (filtrat): {emb_cls1_filtrat.shape[0]:,} prop.")
    print(f"    Distribuție sursă cls1 filtrat:")
    for sursa, n in df_cls1_filtrat["sursa_site"].value_counts().items():
        print(f"      {sursa}: {n:,}")

    # Scoring per propozitie
    sc_cls0 = scor_cosine_max(emb_test, emb_cls0)
    sc_cls1 = scor_cosine_max(emb_test, emb_cls1_filtrat)

    # Agregare la articol
    art = agrega_si_scoreaza(df_test, sc_cls0, sc_cls1)

    # Calcul metrici
    metrici = calcul_metrici_complete(art)

    # Best AUC pe cele 3 teste
    best_A = max(m["auc"] for m in metrici["test_A_cls1_only"].values())
    best_D = max(m["auc"] for m in metrici["test_D_diff"].values())
    print(f"    → Best AUC: Test A = {best_A:.4f}, Test D = {best_D:.4f}")

    return {
        "nume": nume,
        "descriere": descriere,
        "n_corpus_cls0": int(emb_cls0.shape[0]),
        "n_corpus_cls1": int(emb_cls1_filtrat.shape[0]),
        "distributie_sursa_cls1": dict(
            df_cls1_filtrat["sursa_site"].value_counts().items()
        ),
        "metrici": metrici,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestreaza: incarcare → 3 scenarii (baseline + LOSO-V + LOSO-S)."""
    print("=" * 70)
    print("LOSO pe modulul 3 — testul cross-source ultim")
    print("=" * 70)
    seteaza_seed(SEED)

    # ------------------------------------------------------------------
    # Incarcare date
    # ------------------------------------------------------------------
    print("\n[1/5] Încărcare date...")
    df_cls0 = pd.read_parquet(CORPUS_CLS0_PATH)
    df_cls1_full = pd.read_parquet(CORPUS_CLS1_PATH)
    df_test = pd.read_parquet(TEST_SET_PATH)
    print(f"  cls0: {len(df_cls0):,} prop.")
    print(f"  cls1 full: {len(df_cls1_full):,} prop. "
          f"({df_cls1_full['sursa_site'].value_counts().to_dict()})")
    print(f"  test: {len(df_test):,} prop. "
          f"({df_test['articol_id'].nunique()} articole)")

    # ------------------------------------------------------------------
    # Downsample cls1 BASELINE (pentru referinta vs benchmark v4 post-curatare)
    # IMPORTANT: folosim ACELASI seed=42 → acelasi downsample ca in benchmark v4,
    # deci embeddings cache valide.
    # ------------------------------------------------------------------
    print(f"\n[2/5] Downsample cls1 baseline la {DOWNSAMPLE_CLS1_LA:,} (seed=42)...")
    df_cls1_baseline = df_cls1_full.sample(
        n=DOWNSAMPLE_CLS1_LA, random_state=SEED
    ).reset_index(drop=True)
    print(f"  Distribuție baseline: "
          f"{df_cls1_baseline['sursa_site'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Incarcare embeddings din cache (generate de benchmark_v4_post_curatare)
    # ------------------------------------------------------------------
    print("\n[3/5] Încărcare embeddings din cache...")
    try:
        emb_cls0 = incarca_embeddings_cache(
            df_cls0["propozitie"].tolist(), "cls0_corpus"
        )
        emb_cls1_baseline = incarca_embeddings_cache(
            df_cls1_baseline["propozitie"].tolist(),
            "cls1_corpus_v2_downsampled",
        )
        print(f"  cls0 emb: {emb_cls0.shape}")
        print(f"  cls1 baseline emb: {emb_cls1_baseline.shape}")
    except FileNotFoundError as e:
        print(f"  ⚠️ {e}")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Calcul embeddings test set (rapid, ~5 sec pe MPS)
    # ------------------------------------------------------------------
    print("\n[4/5] Calcul embeddings test set...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    emb_test = model.encode(
        df_test["propozitie"].tolist(),
        batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True, device=device,
    )
    print(f"  test emb: {emb_test.shape}")

    # ------------------------------------------------------------------
    # Constructie scenarii LOSO
    # IMPORTANT: filtram la nivel de DataFrame baseline (NU full), apoi
    # filtram embeddings prin INDEX numeric. Asta garanteaza ca folosim
    # exact aceleasi embeddings calculate.
    # ------------------------------------------------------------------
    print("\n[5/5] Rulare scenarii LOSO...")

    # Mask-uri pe corpus baseline
    mask_doar_stopfals = (df_cls1_baseline["sursa_site"] == "stopfals.md").values
    mask_doar_veridica = (df_cls1_baseline["sursa_site"] == "veridica.ro").values

    # LOSO-V: scoatem Veridica → ramane doar Stopfals
    df_loso_v = df_cls1_baseline[mask_doar_stopfals].reset_index(drop=True)
    emb_loso_v = emb_cls1_baseline[mask_doar_stopfals]

    # LOSO-S: scoatem Stopfals → ramane doar Veridica
    df_loso_s = df_cls1_baseline[mask_doar_veridica].reset_index(drop=True)
    emb_loso_s = emb_cls1_baseline[mask_doar_veridica]

    # Sanity check
    assert len(df_loso_v) == emb_loso_v.shape[0], "Mismatch LOSO-V"
    assert len(df_loso_s) == emb_loso_s.shape[0], "Mismatch LOSO-S"
    print(f"  Corpus LOSO-V (doar Stopfals): {len(df_loso_v):,} prop.")
    print(f"  Corpus LOSO-S (doar Veridica): {len(df_loso_s):,} prop.")

    # Rulare 3 scenarii
    rezultate = []

    rezultate.append(ruleaza_scenariu(
        "baseline_standard",
        "Corpus cls1 complet (Veridica + Stopfals downsampled la 5,290) — "
        "referință din benchmark v4 post-curățare.",
        df_cls1_baseline, emb_cls1_baseline,
        df_test, emb_test, emb_cls0,
    ))

    rezultate.append(ruleaza_scenariu(
        "loso_v",
        "LOSO-V: corpus cls1 doar Stopfals (Veridica scoasă). Test articolele "
        "Veridica din test set sunt detectate prin similaritate cu Stopfals?",
        df_loso_v, emb_loso_v,
        df_test, emb_test, emb_cls0,
    ))

    rezultate.append(ruleaza_scenariu(
        "loso_s",
        "LOSO-S: corpus cls1 doar Veridica (Stopfals scos). Simetric cu LOSO-V.",
        df_loso_s, emb_loso_s,
        df_test, emb_test, emb_cls0,
    ))

    # ------------------------------------------------------------------
    # Raport consolidat
    # ------------------------------------------------------------------
    raport_complet = {
        "config": {
            "model": MODEL_NAME,
            "seed": SEED,
            "device": device,
            "varianta": "B (cls0 intact, fără paritate forțată)",
        },
        "test_set": {
            "fisier": str(TEST_SET_PATH),
            "n_articole": int(df_test["articol_id"].nunique()),
            "n_propozitii": len(df_test),
        },
        "scenarii": rezultate,
        "concluzie_finala": _formuleaza_concluzie(rezultate),
    }

    _scrie_raport_json(raport_complet, RAPORT_JSON)
    _scrie_raport_md(raport_complet, RAPORT_MD)

    # Rezumat consola
    print("\n" + "=" * 70)
    print("REZUMAT LOSO MODUL 3")
    print("=" * 70)
    _printeaza_tabel_comparativ(rezultate)
    print()
    print(raport_complet["concluzie_finala"])


def _formuleaza_concluzie(rezultate: list[dict]) -> str:
    """Compara baseline vs LOSO-V vs LOSO-S si formuleaza verdict."""
    by_name = {r["nume"]: r for r in rezultate}

    # Best AUC pe Test D mean (rezultatul nostru principal din benchmark v4)
    base_d_mean = by_name["baseline_standard"]["metrici"]["test_D_diff"]["mean"]["auc"]
    losov_d_mean = by_name["loso_v"]["metrici"]["test_D_diff"]["mean"]["auc"]
    losos_d_mean = by_name["loso_s"]["metrici"]["test_D_diff"]["mean"]["auc"]

    drop_v = base_d_mean - losov_d_mean
    drop_s = base_d_mean - losos_d_mean

    linii = []
    linii.append(f"Test D (diferență cls1−cls0, agregare mean):")
    linii.append(f"  Baseline standard: AUC = {base_d_mean:.4f}")
    linii.append(f"  LOSO-V (corpus = Stopfals only): AUC = {losov_d_mean:.4f} "
                 f"(drop {drop_v:+.4f})")
    linii.append(f"  LOSO-S (corpus = Veridica only): AUC = {losos_d_mean:.4f} "
                 f"(drop {drop_s:+.4f})")
    linii.append("")

    # Comparatie cu modulul 2
    linii.append(f"Comparație cu modulul 2:")
    linii.append(f"  Modul 2 standard (IID): F1 = 100% → recall cls1 = 100%")
    linii.append(f"  Modul 2 LOSO-V: recall cls1 = 29.35% (drop 70.65pp)")
    linii.append("")

    # Verdict
    if losov_d_mean >= 0.85:
        verdict = (
            f"VERDICT: ✓ Modulul 3 (Test D mean) GENERALIZEAZĂ EXCELENT "
            f"cross-source. AUC LOSO-V = {losov_d_mean:.4f} ≥ 0.85, "
            f"drop minim ({drop_v:.4f}). Compensează problema modulului 2. "
            f"Opțiunea 1 din DOSAR_problema_generalizare.md (raportare onestă "
            f"fără remediere modulul 2) devine COMPLET ACCEPTABILĂ."
        )
    elif losov_d_mean >= 0.75:
        verdict = (
            f"VERDICT: ⚠ Modulul 3 generalizează MODERAT cross-source. "
            f"AUC LOSO-V = {losov_d_mean:.4f} (drop {drop_v:+.4f}). "
            f"Mai bun decât modulul 2 (29% recall), dar nu perfect. Sistemul "
            f"combinat oferă o ameliorare reală. Recomandare: raportăm onest "
            f"ambele cifre + drop, poziționăm modulul 3 ca scor complementar."
        )
    elif losov_d_mean >= 0.65:
        verdict = (
            f"VERDICT: ⚠ Modulul 3 are generalizare LIMITATĂ cross-source. "
            f"AUC LOSO-V = {losov_d_mean:.4f}. Nu compensează complet "
            f"modulul 2. Discutăm dacă merită incluse ca scor combinat sau "
            f"ca tool asistiv (Opțiunea C)."
        )
    else:
        verdict = (
            f"VERDICT: ✗ Modulul 3 NU generalizează cross-source. "
            f"AUC LOSO-V = {losov_d_mean:.4f} < 0.65. Problema stylistic "
            f"fingerprint persistă și pe similaritate semantică. "
            f"Trecem la Opțiunea C (tool asistiv vizual) — finding "
            f"metodologic puternic în sine."
        )

    linii.append(verdict)
    return "\n".join(linii)


def _printeaza_tabel_comparativ(rezultate: list[dict]) -> None:
    """Tabel ASCII consola pentru comparatie rapida."""
    print(f"\n{'Scenariu':<22} {'Test A best':>12} {'Test D best':>12} "
          f"{'n_corpus':>10}")
    print("-" * 60)
    for r in rezultate:
        m = r["metrici"]
        best_A = max(x["auc"] for x in m["test_A_cls1_only"].values())
        best_D = max(x["auc"] for x in m["test_D_diff"].values())
        print(f"{r['nume']:<22} {best_A:>12.4f} {best_D:>12.4f} "
              f"{r['n_corpus_cls1']:>10,}")


# ---------------------------------------------------------------------------
# Scriere rapoarte
# ---------------------------------------------------------------------------

def _scrie_raport_json(raport: dict, path: Path) -> None:
    """JSON cu toate datele structurate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(raport, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n  JSON scris: {path}")


def _scrie_raport_md(raport: dict, path: Path) -> None:
    """Raport Markdown lizibil cu tabele de comparatie."""
    config = raport["config"]
    test_set = raport["test_set"]
    scenarii = raport["scenarii"]

    linii = []
    linii.append("# LOSO pe modulul 3 — testul cross-source ultim")
    linii.append("")
    linii.append(
        "Verifică dacă scorul granular cosine generalizează cross-source. "
        "Răspunde la întrebarea-cheie din `DOSAR_problema_generalizare.md`: "
        "modulul 3 compensează stylistic fingerprint-ul modulului 2 "
        "(LOSO-V recall = 29.35%)?"
    )
    linii.append("")

    # Configurare
    linii.append("## Configurare")
    linii.append("")
    linii.append(f"- Model: `{config['model']}`")
    linii.append(f"- Seed: `{config['seed']}`, Device: `{config['device']}`")
    linii.append(f"- Variantă LOSO: **{config['varianta']}**")
    linii.append(f"- Test set: `{test_set['fisier']}` "
                 f"({test_set['n_articole']} articole, "
                 f"{test_set['n_propozitii']:,} prop.)")
    linii.append("")

    # Scenarii
    linii.append("## Scenarii rulate")
    linii.append("")
    for r in scenarii:
        linii.append(f"### `{r['nume']}`")
        linii.append("")
        linii.append(f"_{r['descriere']}_")
        linii.append("")
        linii.append(f"- Corpus cls0: {r['n_corpus_cls0']:,} prop.")
        linii.append(f"- Corpus cls1: {r['n_corpus_cls1']:,} prop.")
        linii.append(f"- Distribuție sursă cls1: {r['distributie_sursa_cls1']}")
        linii.append("")

        # Tabel metrici per scenariu
        m = r["metrici"]
        descrieri_test = {
            "test_A_cls1_only": "Test A — scor_cls1 izolat",
            "test_B_cls0_only": "Test B — scor_cls0 izolat (verificare convenție)",
            "test_D_diff": "Test D — diferență cls1 − cls0 (scor combinat)",
        }
        for test_key, descriere_t in descrieri_test.items():
            linii.append(f"#### {descriere_t}")
            linii.append("")
            linii.append("| Agregare | AUC | Cohen's d | μ(cls0) | "
                         "μ(cls1 V) | μ(cls1 S) | μ(cls1 total) |")
            linii.append("|---|---:|---:|---:|---:|---:|---:|")
            for agr, mt in m[test_key].items():
                linii.append(
                    f"| {agr} | {mt['auc']:.4f} | {mt['cohen_d']:+.4f} | "
                    f"{mt['media_cls0']:.4f} | "
                    f"{mt['media_cls1_veridica']:.4f} | "
                    f"{mt['media_cls1_stopfals']:.4f} | "
                    f"{mt['media_cls1']:.4f} |"
                )
            linii.append("")

    # Tabel comparativ central
    linii.append("## Comparație centrală — Test D mean (rezultatul principal)")
    linii.append("")
    linii.append(
        "Folosim `Test D mean` ca metric principal — robust la artefacte "
        "(vezi `diagnostic_v4.md` pentru justificare)."
    )
    linii.append("")
    linii.append("| Scenariu | n cls1 | AUC | Δ vs baseline | Cohen's d | "
                 "μ(V) | μ(S) |")
    linii.append("|---|---:|---:|---:|---:|---:|---:|")

    base = next(r for r in scenarii if r["nume"] == "baseline_standard")
    base_auc = base["metrici"]["test_D_diff"]["mean"]["auc"]

    for r in scenarii:
        m = r["metrici"]["test_D_diff"]["mean"]
        delta = m["auc"] - base_auc
        delta_str = f"{delta:+.4f}" if r["nume"] != "baseline_standard" else "—"
        linii.append(
            f"| `{r['nume']}` | {r['n_corpus_cls1']:,} | {m['auc']:.4f} | "
            f"{delta_str} | {m['cohen_d']:+.4f} | "
            f"{m['media_cls1_veridica']:.4f} | "
            f"{m['media_cls1_stopfals']:.4f} |"
        )
    linii.append("")

    # Comparatie Test A (cls1 izolat)
    linii.append("## Comparație Test A mean (scor_cls1 izolat)")
    linii.append("")
    linii.append("| Scenariu | AUC | Cohen's d | μ(cls0) | μ(cls1) | "
                 "μ(V) | μ(S) |")
    linii.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in scenarii:
        m = r["metrici"]["test_A_cls1_only"]["mean"]
        linii.append(
            f"| `{r['nume']}` | {m['auc']:.4f} | {m['cohen_d']:+.4f} | "
            f"{m['media_cls0']:.4f} | {m['media_cls1']:.4f} | "
            f"{m['media_cls1_veridica']:.4f} | "
            f"{m['media_cls1_stopfals']:.4f} |"
        )
    linii.append("")

    # Comparatie cu modulul 2
    linii.append("## Comparație cu modulul 2 (clasificator XLM-R)")
    linii.append("")
    linii.append(
        "Modulul 2 (xlmr_baseline_v2): F1 IID standard = 100%, "
        "**recall cls1 LOSO-V = 29.35%** (drop 70.65 puncte procentuale)."
    )
    linii.append("")

    losov = next(r for r in scenarii if r["nume"] == "loso_v")
    losov_auc = losov["metrici"]["test_D_diff"]["mean"]["auc"]
    drop_v = base_auc - losov_auc

    linii.append("| Modul | Standard | LOSO-V | Drop |")
    linii.append("|---|---:|---:|---:|")
    linii.append(f"| Modul 2 (recall cls1) | 100% | 29.35% | "
                 f"−70.65pp |")
    linii.append(f"| Modul 3 (Test D AUC mean) | {base_auc:.4f} | "
                 f"{losov_auc:.4f} | {-drop_v:+.4f} |")
    linii.append("")

    # Concluzie finala
    linii.append("## Concluzie finală")
    linii.append("")
    linii.append("```")
    linii.append(raport["concluzie_finala"])
    linii.append("```")
    linii.append("")

    # Referinta viitoare pentru teza
    linii.append("## Implicații pentru teză (capitolul „Evaluare cross-source")
    linii.append("")
    if losov_auc >= 0.75:
        linii.append(
            "- **Finding 7 (proaspăt):** Modulul 3 generalizează cross-source "
            f"semnificativ mai bine decât modulul 2 (AUC LOSO-V = "
            f"{losov_auc:.4f} pe modul 3 vs recall 29% pe modul 2). "
            "Confirmă ipoteza că similaritatea semantică e mai puțin "
            "vulnerabilă la stylistic fingerprint decât clasificarea."
        )
        linii.append("")
        linii.append(
            "- **Decizie metodologică:** Opțiunea 1 din "
            "`DOSAR_problema_generalizare.md` (raportare onestă fără remediere "
            "modulul 2) devine acceptabilă — modulul 3 oferă răspunsul "
            "complementar la limitarea modulului 2."
        )
    else:
        linii.append(
            "- **Finding 7 (proaspăt):** Stylistic fingerprint persistă și pe "
            f"similaritate semantică (AUC LOSO-V modul 3 = {losov_auc:.4f}). "
            "Confirmare suplimentară a problemei structurale documentate "
            "în literatura românească pe fake news detection."
        )
    linii.append("")
    linii.append("---")
    linii.append("")
    linii.append("*Modul 3 · Pasul A3 · LOSO cross-source*")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(linii))
    print(f"  MD scris: {path}")


if __name__ == "__main__":
    main()
