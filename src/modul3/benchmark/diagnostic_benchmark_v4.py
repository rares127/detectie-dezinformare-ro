"""
Diagnostic benchmark v4 — verificare 4 probe pentru validarea AUC-urilor mari.

Context:
--------
Benchmark v4 a raportat rezultate surprinzator de mari:
  - Test A (cls1-only) min = 0.9774 AUC
  - Test B (cls0-only) min = 0.9435 AUC  ← SUSPECT: corpus cls0 neatins de la v3
  - Test D (diferenta) mean = 0.9739 AUC

In v3, pe acest corpus cls0, AUC-ul era doar 0.552 (mean/max). Saltul la
0.9435 prin schimbarea agregarii (mean → min) e un steag rosu care sugereaza
ca agregarea `min` capteaza o proprietate structurala (lungime articol,
propozitii atipice) mai degraba decat similaritate semantica reala.

Obiectiv:
---------
Separam "semnal real" de "artefact de agregare" prin 4 probe diagnostic:

  1. Distributia lungimii articolelor pe cls0 vs cls1 (structural bias?)
  2. AUC pe subset balansat pe lungime + corelatie nr_prop × scor_min
  3. Inspectie calitativa propozitii "min" per articol (10 cls0 + 10 cls1)
  4. Agregare top-k mean (k=5) — robusta fata de extreme

Output:
  - findings/diagnostic_v4.md + .json

Utilizare:
  python scripts/diagnostic_benchmark_v4.py
"""

from __future__ import annotations

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
TEST_SET_PATH = Path("data/processed/subset_benchmark_v3.parquet")

# Cache embeddings din benchmark_v4 (reutilizate integral)
CACHE_DIR = Path("data/processed/embeddings_cache")

RAPORT_MD = Path("findings/diagnostic_v4.md")
RAPORT_JSON = Path("findings/diagnostic_v4.json")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SEED = 42
DOWNSAMPLE_CLS1_LA = 5_290

# Nr. exemple calitative per clasa pentru Proba 3
N_EXEMPLE_PER_CLASA = 10

# top-k pentru Proba 4 (mean al top-k similaritati per propozitie)
TOP_K = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seteaza_seed(seed: int = SEED) -> None:
    """Fixeaza seed-ul pentru reproductibilitate."""
    random.seed(seed)
    np.random.seed(seed)


def calculeaza_hash_corpus(texts, model_name: str) -> str:
    """Recalculeaza hash identic cu cel din benchmark_v4 — pentru a gasi cache."""
    import hashlib
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\n")
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def incarca_embeddings_din_cache(texts: list[str], nume: str) -> np.ndarray:
    """Incarca embeddings din cache. Arunca eroare daca nu exista."""
    h = calculeaza_hash_corpus(texts, MODEL_NAME)
    cache_path = CACHE_DIR / f"{nume}_{h}.npy"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache lipsă: {cache_path}. "
            f"Rulează benchmark_v4.py mai întâi ca să generezi embeddings."
        )
    emb = np.load(cache_path)
    assert emb.shape[0] == len(texts), (
        f"Cache corupt: {emb.shape[0]} vs {len(texts)} texte"
    )
    return emb


def scor_cosine_max(emb_test: np.ndarray, emb_corpus: np.ndarray,
                    batch: int = 256) -> np.ndarray:
    """Cosine max per propozitie test vs corpus (embeddings normalizate L2)."""
    n = emb_test.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch):
        b = emb_test[i:i + batch]
        out[i:i + batch] = (b @ emb_corpus.T).max(axis=1)
    return out


def scor_cosine_topk_mean(emb_test: np.ndarray, emb_corpus: np.ndarray,
                          k: int = TOP_K, batch: int = 256) -> np.ndarray:
    """Media top-k similaritati per propozitie test (mai robusta decat max)."""
    n = emb_test.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch):
        b = emb_test[i:i + batch]
        sim = b @ emb_corpus.T  # (b_size, n_corpus)
        # Top-k pe fiecare rand — folosim np.partition pentru eficienta
        # argpartition aduce cele mai mari k in ultimele k pozitii
        topk = np.partition(sim, -k, axis=1)[:, -k:]
        out[i:i + batch] = topk.mean(axis=1)
    return out


def auc(labels: np.ndarray, scoruri: np.ndarray) -> float:
    """AUC-ROC cu handling pentru single-class."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scoruri))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Marimea efectului (b − a) / std_pooled."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_pooled = ((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
    var_pooled /= (len(a) + len(b) - 2)
    if var_pooled <= 0:
        return float("nan")
    return float((b.mean() - a.mean()) / np.sqrt(var_pooled))


# ---------------------------------------------------------------------------
# Proba 1 — distributia lungimii articolelor
# ---------------------------------------------------------------------------

def proba_1_distributie_lungime(df_test: pd.DataFrame) -> dict:
    """
    Calculeaza distributia numarului de propozitii per articol, stratificata
    pe label si sursa.

    Interpretare:
      - Daca cls0 si cls1 au distributii sistematic diferite (ex. cls0 cu
        mai multe propozitii), atunci `min` pe cls0 va fi biased structural
        (cu cat ai mai multe propozitii, cu atat e mai probabil sa ai una
        cu scor scazut).
    """
    print("\n" + "=" * 70)
    print("PROBA 1: Distribuția lungimii articolelor (nr. propoziții)")
    print("=" * 70)

    # nr_prop per articol
    per_art = df_test.groupby(
        ["articol_id", "label_numeric", "sursa_site"]
    ).size().reset_index(name="nr_prop")

    # Statistici per label
    stats_label = {}
    for lbl in (0, 1):
        sub = per_art[per_art["label_numeric"] == lbl]["nr_prop"]
        stats_label[f"cls{lbl}"] = {
            "n_articole": int(len(sub)),
            "min": int(sub.min()),
            "p25": float(sub.quantile(0.25)),
            "mediana": float(sub.median()),
            "p75": float(sub.quantile(0.75)),
            "max": int(sub.max()),
            "media": float(sub.mean()),
            "std": float(sub.std()),
        }

    # Statistici per sursa (in interiorul label-urilor)
    stats_sursa = {}
    for sursa, sub in per_art.groupby("sursa_site"):
        stats_sursa[sursa] = {
            "n_articole": int(len(sub)),
            "mediana": float(sub["nr_prop"].median()),
            "media": float(sub["nr_prop"].mean()),
            "min": int(sub["nr_prop"].min()),
            "max": int(sub["nr_prop"].max()),
        }

    # Printez tabel consola
    print(f"\nDistribuție per label:")
    print(f"  {'':5s} {'n':>4s}  {'min':>4s} {'p25':>5s} {'med':>5s} "
          f"{'p75':>5s} {'max':>4s}  {'media':>6s} {'std':>5s}")
    for lbl_key, s in stats_label.items():
        print(f"  {lbl_key:<5s} {s['n_articole']:>4d}  "
              f"{s['min']:>4d} {s['p25']:>5.1f} {s['mediana']:>5.1f} "
              f"{s['p75']:>5.1f} {s['max']:>4d}  {s['media']:>6.2f} "
              f"{s['std']:>5.2f}")

    print(f"\nDistribuție per sursă:")
    for sursa, s in sorted(stats_sursa.items()):
        print(f"  {sursa:<20s} n={s['n_articole']:>3d}  "
              f"mediana={s['mediana']:>5.1f}  "
              f"media={s['media']:>6.2f}  "
              f"range=[{s['min']}, {s['max']}]")

    # Interpretare automata
    cls0_med = stats_label["cls0"]["mediana"]
    cls1_med = stats_label["cls1"]["mediana"]
    ratio = cls0_med / cls1_med if cls1_med > 0 else float("inf")

    print(f"\nMediana cls0 / mediana cls1 = {cls0_med:.1f} / {cls1_med:.1f} "
          f"= {ratio:.2f}x")

    if ratio > 1.5 or ratio < 0.67:
        interpretare = (
            f"ATENȚIE: diferență mare între mediane (raport {ratio:.2f}x). "
            f"Artefact de lungime PLAUZIBIL — agregarea min penalizează "
            f"articolele mai lungi."
        )
    else:
        interpretare = (
            f"Distribuții similare (raport {ratio:.2f}x în intervalul [0.67, 1.5]). "
            f"Artefact de lungime puțin probabil."
        )

    print(f"\n→ {interpretare}")

    return {
        "stats_label": stats_label,
        "stats_sursa": stats_sursa,
        "raport_mediane_cls0_cls1": float(ratio),
        "interpretare": interpretare,
    }


# ---------------------------------------------------------------------------
# Proba 2 — AUC pe subset balansat + corelatie nr_prop × scor
# ---------------------------------------------------------------------------

def proba_2_subset_balansat_corelatie(
    df_test: pd.DataFrame,
    scoruri_propozitii_cls1: np.ndarray,
    scoruri_propozitii_cls0: np.ndarray,
) -> dict:
    """
    Doua sub-analize:

    A. AUC pe subset balansat pe lungime (doar articole cu nr_prop ∈ [p25, p75]
       combinat pe ambele clase) → daca AUC scade semnificativ, diferenta
       de lungime era motorul.

    B. Corelatia Pearson intre nr_prop si scor_min per clasa. Daca e
       puternic negativa pe cls0 (sau cls1), atunci min e dominat de
       lungime, nu de continut.
    """
    print("\n" + "=" * 70)
    print("PROBA 2: AUC pe subset balansat + corelație lungime × scor")
    print("=" * 70)

    df = df_test.copy()
    df["scor_cls0_prop"] = scoruri_propozitii_cls0
    df["scor_cls1_prop"] = scoruri_propozitii_cls1

    # Agregare min per articol
    per_art = df.groupby(
        ["articol_id", "label_numeric", "sursa_site"]
    ).agg(
        nr_prop=("propozitie", "count"),
        scor_cls1_min=("scor_cls1_prop", "min"),
        scor_cls0_min=("scor_cls0_prop", "min"),
        scor_cls1_mean=("scor_cls1_prop", "mean"),
        scor_cls0_mean=("scor_cls0_prop", "mean"),
    ).reset_index()

    # diff
    per_art["diff_min"] = per_art["scor_cls1_min"] - per_art["scor_cls0_min"]
    per_art["diff_mean"] = per_art["scor_cls1_mean"] - per_art["scor_cls0_mean"]

    # --- A. Subset balansat pe lungime ---
    # Determinam intervalul [p25, p75] combinat pe ambele clase
    p25 = per_art["nr_prop"].quantile(0.25)
    p75 = per_art["nr_prop"].quantile(0.75)

    mask_balansat = (per_art["nr_prop"] >= p25) & (per_art["nr_prop"] <= p75)
    subset = per_art[mask_balansat]

    print(f"\nInterval balansat [p25, p75] = [{p25:.0f}, {p75:.0f}] propoziții/art.")
    print(f"Articole în subset: {len(subset)} "
          f"({(subset['label_numeric']==0).sum()} cls0 + "
          f"{(subset['label_numeric']==1).sum()} cls1)")

    labels_sub = subset["label_numeric"].values

    auc_full = {}
    auc_sub = {}
    for coloana in ("scor_cls1_min", "scor_cls0_min", "diff_min",
                    "scor_cls1_mean", "scor_cls0_mean", "diff_mean"):
        auc_full[coloana] = auc(
            per_art["label_numeric"].values,
            per_art[coloana].values,
        )
        if len(np.unique(labels_sub)) < 2:
            auc_sub[coloana] = float("nan")
        else:
            auc_sub[coloana] = auc(labels_sub, subset[coloana].values)

    print(f"\nComparație AUC full (n=167) vs subset balansat (n={len(subset)}):")
    print(f"  {'Coloana':<22s} {'full':>8s}  {'subset':>8s}  {'Δ':>7s}")
    for c in auc_full:
        delta = auc_sub[c] - auc_full[c]
        print(f"  {c:<22s} {auc_full[c]:>8.4f}  {auc_sub[c]:>8.4f}  "
              f"{delta:+7.4f}")

    # --- B. Corelatie lungime × scor ---
    print(f"\nCorelații Pearson (nr_prop × scor), per clasă:")
    correlatii = {}
    for lbl in (0, 1):
        sub_lbl = per_art[per_art["label_numeric"] == lbl]
        cors = {}
        for col in ("scor_cls1_min", "scor_cls0_min", "diff_min",
                    "scor_cls1_mean", "scor_cls0_mean", "diff_mean"):
            if len(sub_lbl) < 3:
                cors[col] = float("nan")
            else:
                r = sub_lbl[["nr_prop", col]].corr().iloc[0, 1]
                cors[col] = float(r)
        correlatii[f"cls{lbl}"] = cors
        print(f"\n  cls{lbl} (n={len(sub_lbl)}):")
        for col, r in cors.items():
            strong = "⚠️ PUTERNICĂ" if abs(r) > 0.5 else ""
            print(f"    {col:<22s} r = {r:+.4f}  {strong}")

    # Interpretare automata
    interpretari = []

    # Daca AUC scade semnificativ pe subset → artefact lungime
    delta_cls1_min = auc_sub["scor_cls1_min"] - auc_full["scor_cls1_min"]
    delta_diff_mean = auc_sub["diff_mean"] - auc_full["diff_mean"]
    if delta_cls1_min < -0.10:
        interpretari.append(
            f"AUC Test A (min) scade cu {delta_cls1_min:+.3f} pe subset "
            f"balansat → artefact lungime CONFIRMAT pentru agregarea min."
        )
    elif delta_cls1_min < -0.05:
        interpretari.append(
            f"AUC Test A (min) scade cu {delta_cls1_min:+.3f} → artefact "
            f"lungime PROBABIL pentru min."
        )
    else:
        interpretari.append(
            f"AUC Test A (min) scade doar cu {delta_cls1_min:+.3f} pe subset "
            f"→ semnalul NU e artefact de lungime."
        )

    if abs(delta_diff_mean) < 0.05:
        interpretari.append(
            f"AUC Test D (mean) stabil pe subset ({delta_diff_mean:+.3f}) → "
            f"agregarea mean pe diferență e robustă la lungime."
        )

    # Corelatie puternica intre nr_prop si scor_min
    cor_cls0_min = correlatii["cls0"]["scor_cls1_min"]
    if abs(cor_cls0_min) > 0.5:
        interpretari.append(
            f"Corelație puternică |r|={abs(cor_cls0_min):.2f} între "
            f"nr_prop și scor_cls1_min pe cls0 → scorul min e dominat de "
            f"structura articolului pe cls0."
        )

    print("\n→ " + "\n→ ".join(interpretari))

    return {
        "interval_balansat_p25_p75": [float(p25), float(p75)],
        "n_articole_subset": int(len(subset)),
        "n_cls0_subset": int((subset["label_numeric"] == 0).sum()),
        "n_cls1_subset": int((subset["label_numeric"] == 1).sum()),
        "auc_full": auc_full,
        "auc_subset_balansat": auc_sub,
        "delta_auc_full_vs_subset": {
            c: auc_sub[c] - auc_full[c] for c in auc_full
        },
        "corelatii_nrprop_scor": correlatii,
        "interpretari": interpretari,
    }


# ---------------------------------------------------------------------------
# Proba 3 — inspectie calitativa propozitii min
# ---------------------------------------------------------------------------

def proba_3_inspectie_calitativa(
    df_test: pd.DataFrame,
    scoruri_propozitii_cls1: np.ndarray,
    scoruri_propozitii_cls0: np.ndarray,
    df_cls1: pd.DataFrame,
    emb_test: np.ndarray,
    emb_cls1: np.ndarray,
) -> dict:
    """
    Pentru 10 articole cls0 + 10 cls1 (aleatoare), afiseaza:
      - Propozitia care a dat scorul cls1 min din articol
      - Propozitia cea mai similara din corpusul cls1 (pentru context)

    Interpretare: daca propozitiile "min" de pe cls0 sunt sistematic structurale
    (titluri, capturi, note redactionale) care nu apar in corpus, atunci
    scorul min nu masoara propaganda, masoara "difera de presa standarda".
    """
    print("\n" + "=" * 70)
    print("PROBA 3: Inspecție calitativă propoziții min "
          f"({N_EXEMPLE_PER_CLASA} per clasă)")
    print("=" * 70)

    seteaza_seed(SEED)

    df = df_test.copy().reset_index(drop=True)
    df["scor_cls1_prop"] = scoruri_propozitii_cls1
    df["scor_cls0_prop"] = scoruri_propozitii_cls0
    df["idx_test"] = df.index  # pozitia in emb_test

    # Selectez exemple aleatoare per clasa
    articole_cls0 = df["articol_id"][df["label_numeric"] == 0].unique()
    articole_cls1 = df["articol_id"][df["label_numeric"] == 1].unique()

    rng = np.random.default_rng(SEED)
    ex_cls0 = rng.choice(articole_cls0, size=min(N_EXEMPLE_PER_CLASA, len(articole_cls0)),
                         replace=False)
    ex_cls1 = rng.choice(articole_cls1, size=min(N_EXEMPLE_PER_CLASA, len(articole_cls1)),
                         replace=False)

    corpus_cls1_texts = df_cls1["propozitie"].tolist()

    exemple = {"cls0": [], "cls1": []}

    for lbl_key, articole in (("cls0", ex_cls0), ("cls1", ex_cls1)):
        for aid in articole:
            art_prop = df[df["articol_id"] == aid]
            # Identificam randul cu scor_cls1 MINIM (= cel care dicteaza min-ul)
            idx_min_local = art_prop["scor_cls1_prop"].idxmin()
            prop_min = art_prop.loc[idx_min_local]

            # Gasim propozitia din corpus cls1 cu similaritate max cu prop_min
            idx_test_in_emb = int(prop_min["idx_test"])
            sim_cu_corpus = emb_test[idx_test_in_emb] @ emb_cls1.T
            idx_match_corpus = int(sim_cu_corpus.argmax())
            scor_match = float(sim_cu_corpus[idx_match_corpus])
            prop_match = corpus_cls1_texts[idx_match_corpus]

            exemple[lbl_key].append({
                "articol_id": str(aid),
                "sursa": str(prop_min["sursa_site"]),
                "nr_prop_articol": int(len(art_prop)),
                "propozitia_min": str(prop_min["propozitie"]),
                "nr_cuvinte": int(prop_min["nr_cuvinte"]),
                "scor_min": float(prop_min["scor_cls1_prop"]),
                "match_corpus_cls1": prop_match,
                "match_scor": scor_match,
            })

    # Print exemple consola
    for lbl_key in ("cls0", "cls1"):
        print(f"\n--- {lbl_key.upper()} (10 articole aleatoare) ---")
        for i, e in enumerate(exemple[lbl_key], 1):
            print(f"\n  [{i}] {e['articol_id']} · {e['sursa']} · "
                  f"{e['nr_prop_articol']} prop/art")
            print(f"      Prop. min ({e['nr_cuvinte']}w, scor={e['scor_min']:.3f}):")
            print(f"      → {e['propozitia_min'][:180]}")
            print(f"      Match corpus cls1 (scor={e['match_scor']:.3f}):")
            print(f"      → {e['match_corpus_cls1'][:180]}")

    # Statistici agregate: lungime medie propozitii min
    cls0_lungimi = [e["nr_cuvinte"] for e in exemple["cls0"]]
    cls1_lungimi = [e["nr_cuvinte"] for e in exemple["cls1"]]
    cls0_scoruri = [e["scor_min"] for e in exemple["cls0"]]
    cls1_scoruri = [e["scor_min"] for e in exemple["cls1"]]

    print(f"\nStatistici propoziții min (sample {N_EXEMPLE_PER_CLASA} per clasă):")
    print(f"  cls0: lungime medie = {np.mean(cls0_lungimi):.1f}w, "
          f"scor med = {np.mean(cls0_scoruri):.3f}")
    print(f"  cls1: lungime medie = {np.mean(cls1_lungimi):.1f}w, "
          f"scor med = {np.mean(cls1_scoruri):.3f}")

    return {
        "exemple": exemple,
        "stats_sample": {
            "cls0_lungime_medie_propmin": float(np.mean(cls0_lungimi)),
            "cls0_scor_mediu_min": float(np.mean(cls0_scoruri)),
            "cls1_lungime_medie_propmin": float(np.mean(cls1_lungimi)),
            "cls1_scor_mediu_min": float(np.mean(cls1_scoruri)),
        },
    }


# ---------------------------------------------------------------------------
# Proba 4 — top-k mean (k=5)
# ---------------------------------------------------------------------------

def proba_4_topk_mean(
    df_test: pd.DataFrame,
    emb_test: np.ndarray,
    emb_cls0: np.ndarray,
    emb_cls1: np.ndarray,
) -> dict:
    """
    Recalculeaza scorurile cu top-k mean (k=5) in loc de max per propozitie.

    Top-k mean e mai robust la extreme: in loc sa ia doar cea mai mare
    similaritate (care poate fi accidentala), media top-5 cere consistenta.
    """
    print("\n" + "=" * 70)
    print(f"PROBA 4: Agregare top-{TOP_K} mean (robustă la extreme)")
    print("=" * 70)

    # Scoruri propozitie cu top-k mean
    sc_cls0 = scor_cosine_topk_mean(emb_test, emb_cls0, k=TOP_K)
    sc_cls1 = scor_cosine_topk_mean(emb_test, emb_cls1, k=TOP_K)

    df = df_test.copy()
    df["scor_cls0_prop"] = sc_cls0
    df["scor_cls1_prop"] = sc_cls1

    def _p10(s):
        return float(np.percentile(s, 10))

    per_art = df.groupby(
        ["articol_id", "label_numeric"]
    ).agg(
        scor_cls1_mean=("scor_cls1_prop", "mean"),
        scor_cls1_min=("scor_cls1_prop", "min"),
        scor_cls1_p10=("scor_cls1_prop", _p10),
        scor_cls0_mean=("scor_cls0_prop", "mean"),
        scor_cls0_min=("scor_cls0_prop", "min"),
        scor_cls0_p10=("scor_cls0_prop", _p10),
    ).reset_index()

    per_art["diff_mean"] = per_art["scor_cls1_mean"] - per_art["scor_cls0_mean"]
    per_art["diff_min"] = per_art["scor_cls1_min"] - per_art["scor_cls0_min"]
    per_art["diff_p10"] = per_art["scor_cls1_p10"] - per_art["scor_cls0_p10"]

    labels = per_art["label_numeric"].values

    # AUC pentru toate combinatiile
    rezultate = {}
    for test_name, cols in {
        "test_A_cls1_only": ["scor_cls1_mean", "scor_cls1_min", "scor_cls1_p10"],
        "test_B_cls0_only": ["scor_cls0_mean", "scor_cls0_min", "scor_cls0_p10"],
        "test_D_diff": ["diff_mean", "diff_min", "diff_p10"],
    }.items():
        rezultate[test_name] = {}
        for c in cols:
            agr = c.split("_")[-1]
            rezultate[test_name][agr] = {
                "auc": auc(labels, per_art[c].values),
                "media_cls0": float(per_art[per_art["label_numeric"] == 0][c].mean()),
                "media_cls1": float(per_art[per_art["label_numeric"] == 1][c].mean()),
            }

    # Print tabel
    print(f"\nRezultate cu top-{TOP_K} mean per propoziție:")
    print(f"  {'Test':<25s} {'Agregare':<10s} {'AUC (top-k)':>12s}")
    for tn, rez in rezultate.items():
        for agr, m in rez.items():
            print(f"  {tn:<25s} {agr:<10s} {m['auc']:>12.4f}")

    return {
        "top_k": TOP_K,
        "rezultate": rezultate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestreaza cele 4 probe si scrie raport consolidat."""
    print("=" * 70)
    print("DIAGNOSTIC BENCHMARK v4 — 4 probe")
    print("=" * 70)
    seteaza_seed(SEED)

    # Incarcare date
    print("\n[setup] Încărcare date...")
    df_cls0 = pd.read_parquet(CORPUS_CLS0_PATH)
    df_cls1_full = pd.read_parquet(CORPUS_CLS1_PATH)
    df_test = pd.read_parquet(TEST_SET_PATH)

    # Downsample cls1 cu acelasi seed → acelasi subset ca in benchmark_v4
    df_cls1 = df_cls1_full.sample(
        n=DOWNSAMPLE_CLS1_LA, random_state=SEED
    ).reset_index(drop=True)

    print(f"  cls0: {len(df_cls0):,} prop.")
    print(f"  cls1 (downsampled, seed={SEED}): {len(df_cls1):,} prop.")
    print(f"  test: {len(df_test):,} prop. ({df_test['articol_id'].nunique()} art.)")

    # Incarcare embeddings din cache (generate de benchmark_v4)
    print("\n[setup] Încărcare embeddings din cache...")
    try:
        emb_cls0 = incarca_embeddings_din_cache(
            df_cls0["propozitie"].tolist(), "cls0_corpus"
        )
        emb_cls1 = incarca_embeddings_din_cache(
            df_cls1["propozitie"].tolist(), "cls1_corpus_v2_downsampled"
        )
        print(f"  cls0 embeddings: {emb_cls0.shape}")
        print(f"  cls1 embeddings: {emb_cls1.shape}")
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        print("  Recalculez embeddings pe loc (va dura)...")
        from sentence_transformers import SentenceTransformer
        device = "mps"  # presupunem MPS conform benchmark_v4
        model = SentenceTransformer(MODEL_NAME, device=device)
        emb_cls0 = model.encode(
            df_cls0["propozitie"].tolist(),
            batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True, device=device,
        )
        emb_cls1 = model.encode(
            df_cls1["propozitie"].tolist(),
            batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True, device=device,
        )

    # Embeddings test — nu sunt cache-uite in benchmark_v4, recalculam rapid
    # (2,181 prop. × ~400 prop/s pe MPS = ~5 secunde)
    print("\n[setup] Calcul embeddings test set (nu e cache-uit)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="mps")
    emb_test = model.encode(
        df_test["propozitie"].tolist(),
        batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True, device="mps",
    )
    print(f"  test embeddings: {emb_test.shape}")

    # Scoruri propozitie (max per propozitie) — reutilizate intre probe
    print("\n[setup] Scoruri propoziție (cosine max)...")
    sc_cls0_prop = scor_cosine_max(emb_test, emb_cls0)
    sc_cls1_prop = scor_cosine_max(emb_test, emb_cls1)

    # ------------------------------------------------------------------
    # Rulare probe
    # ------------------------------------------------------------------
    rez1 = proba_1_distributie_lungime(df_test)
    rez2 = proba_2_subset_balansat_corelatie(df_test, sc_cls1_prop, sc_cls0_prop)
    rez3 = proba_3_inspectie_calitativa(
        df_test, sc_cls1_prop, sc_cls0_prop, df_cls1, emb_test, emb_cls1
    )
    rez4 = proba_4_topk_mean(df_test, emb_test, emb_cls0, emb_cls1)

    # ------------------------------------------------------------------
    # Concluzii finale
    # ------------------------------------------------------------------
    concluzii = _formuleaza_concluzii_finale(rez1, rez2, rez3, rez4)

    print("\n" + "=" * 70)
    print("CONCLUZII DIAGNOSTIC")
    print("=" * 70)
    for linie in concluzii:
        print(f"  • {linie}")

    # ------------------------------------------------------------------
    # Scriere rapoarte
    # ------------------------------------------------------------------
    raport = {
        "proba_1_lungime": rez1,
        "proba_2_subset_corelatie": rez2,
        "proba_3_inspectie": rez3,
        "proba_4_topk_mean": rez4,
        "concluzii": concluzii,
    }

    RAPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RAPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(raport, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n  JSON scris: {RAPORT_JSON}")

    _scrie_raport_md(raport, RAPORT_MD)
    print(f"  MD scris: {RAPORT_MD}")


def _formuleaza_concluzii_finale(rez1, rez2, rez3, rez4) -> list[str]:
    """Sintetizeaza cele 4 probe intr-un verdict global."""
    concluzii = []

    # P1: lungime
    raport_med = rez1["raport_mediane_cls0_cls1"]
    concluzii.append(
        f"P1 · Raport mediane cls0/cls1 = {raport_med:.2f}x "
        f"→ {'DIFERENȚĂ STRUCTURALĂ' if (raport_med > 1.5 or raport_med < 0.67) else 'distribuții comparabile'}"
    )

    # P2: drop AUC pe subset balansat
    delta_min = rez2["delta_auc_full_vs_subset"]["scor_cls1_min"]
    delta_diff = rez2["delta_auc_full_vs_subset"]["diff_mean"]
    concluzii.append(
        f"P2 · ΔAUC Test A min (subset balansat): {delta_min:+.4f} "
        f"→ {'ARTEFACT LUNGIME CONFIRMAT' if delta_min < -0.10 else 'semnal robust la lungime'}"
    )
    concluzii.append(
        f"P2 · ΔAUC Test D mean (subset balansat): {delta_diff:+.4f} "
        f"→ {'agregarea mean stabilă' if abs(delta_diff) < 0.05 else 'agregarea mean afectată de lungime'}"
    )

    # P2: corelatie
    cor_cls0_min = rez2["corelatii_nrprop_scor"]["cls0"]["scor_cls1_min"]
    if abs(cor_cls0_min) > 0.5:
        concluzii.append(
            f"P2 · Corelație nr_prop × scor_cls1_min pe cls0: "
            f"r={cor_cls0_min:+.3f} → min dominat de lungime"
        )

    # P3: diferenta lungime propozitii min
    cls0_l = rez3["stats_sample"]["cls0_lungime_medie_propmin"]
    cls1_l = rez3["stats_sample"]["cls1_lungime_medie_propmin"]
    concluzii.append(
        f"P3 · Lungime medie prop. min: cls0={cls0_l:.1f}w vs cls1={cls1_l:.1f}w"
    )

    # P4: AUC top-k mean vs max
    top_A = max(rez4["rezultate"]["test_A_cls1_only"][a]["auc"]
                for a in ("mean", "min", "p10"))
    top_D = max(rez4["rezultate"]["test_D_diff"][a]["auc"]
                for a in ("mean", "min", "p10"))
    concluzii.append(
        f"P4 · Best AUC top-{TOP_K} mean: Test A = {top_A:.4f}, Test D = {top_D:.4f}"
    )

    # Verdict global
    # Criterii:
    #   - daca ΔAUC min < -0.10 → artefact confirmat
    #   - daca ΔAUC mean < 0.05 SI top-k AUC > 0.75 → semnal real
    if delta_min < -0.10:
        if top_A >= 0.75:
            verdict = (
                "VERDICT: min e ARTEFACT (ΔAUC drop mare), DAR semnalul real "
                f"există (top-k AUC = {top_A:.4f} ≥ 0.75). Opțiunea A "
                "rămâne validată, raportăm pe top-k mean sau mean, nu min."
            )
        else:
            verdict = (
                "VERDICT: min e artefact, top-k mean nu mai separă → Opțiunea "
                "A NU e viabilă. Pariul inițial pe rezultatul mare era fals."
            )
    else:
        verdict = (
            "VERDICT: Min e semnal REAL (ΔAUC stabil pe subset balansat). "
            "AUC 0.98 din benchmark v4 rămâne valid. Raportăm pe min."
        )
    concluzii.append(verdict)

    return concluzii


def _scrie_raport_md(raport: dict, path: Path) -> None:
    """Raport Markdown pentru citire umana."""
    linii = []
    linii.append("# Diagnostic Benchmark v4 — 4 probe")
    linii.append("")
    linii.append(
        "Verifică dacă AUC-urile mari din benchmark v4 (0.98 pe Test A min, "
        "0.94 pe Test B min) sunt semnal real sau artefact metodologic "
        "(în special: artefact de lungime articol)."
    )
    linii.append("")

    # Proba 1
    linii.append("## Proba 1 — Distribuția lungimii articolelor")
    linii.append("")
    p1 = raport["proba_1_lungime"]
    linii.append(f"**Raport mediane cls0 / cls1 = {p1['raport_mediane_cls0_cls1']:.2f}x**")
    linii.append("")
    linii.append(f"_{p1['interpretare']}_")
    linii.append("")
    linii.append("| Clasă | n | min | p25 | mediana | p75 | max | media | std |")
    linii.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, s in p1["stats_label"].items():
        linii.append(
            f"| {k} | {s['n_articole']} | {s['min']} | {s['p25']:.1f} | "
            f"{s['mediana']:.1f} | {s['p75']:.1f} | {s['max']} | "
            f"{s['media']:.2f} | {s['std']:.2f} |"
        )
    linii.append("")

    linii.append("### Per sursă")
    linii.append("")
    linii.append("| Sursă | n | min | mediana | media | max |")
    linii.append("|---|---:|---:|---:|---:|---:|")
    for sursa, s in sorted(p1["stats_sursa"].items()):
        linii.append(
            f"| {sursa} | {s['n_articole']} | {s['min']} | "
            f"{s['mediana']:.1f} | {s['media']:.2f} | {s['max']} |"
        )
    linii.append("")

    # Proba 2
    linii.append("## Proba 2 — AUC subset balansat + corelație lungime × scor")
    linii.append("")
    p2 = raport["proba_2_subset_corelatie"]
    linii.append(
        f"**Interval balansat [p25, p75]:** "
        f"[{p2['interval_balansat_p25_p75'][0]:.0f}, "
        f"{p2['interval_balansat_p25_p75'][1]:.0f}] propoziții/articol. "
        f"**n={p2['n_articole_subset']}** ({p2['n_cls0_subset']} cls0 + "
        f"{p2['n_cls1_subset']} cls1)."
    )
    linii.append("")
    linii.append("### AUC comparație: full vs subset balansat")
    linii.append("")
    linii.append("| Coloană | AUC full (n=167) | AUC subset | Δ |")
    linii.append("|---|---:|---:|---:|")
    for c, d in p2["delta_auc_full_vs_subset"].items():
        linii.append(
            f"| `{c}` | {p2['auc_full'][c]:.4f} | "
            f"{p2['auc_subset_balansat'][c]:.4f} | {d:+.4f} |"
        )
    linii.append("")

    linii.append("### Corelații Pearson (nr_prop × scor) per clasă")
    linii.append("")
    linii.append("| Coloană | r pe cls0 | r pe cls1 |")
    linii.append("|---|---:|---:|")
    for col in p2["corelatii_nrprop_scor"]["cls0"]:
        r0 = p2["corelatii_nrprop_scor"]["cls0"][col]
        r1 = p2["corelatii_nrprop_scor"]["cls1"][col]
        linii.append(f"| `{col}` | {r0:+.4f} | {r1:+.4f} |")
    linii.append("")

    linii.append("### Interpretări Proba 2")
    linii.append("")
    for interp in p2["interpretari"]:
        linii.append(f"- {interp}")
    linii.append("")

    # Proba 3
    linii.append("## Proba 3 — Inspecție calitativă propoziții min")
    linii.append("")
    p3 = raport["proba_3_inspectie"]
    s3 = p3["stats_sample"]
    linii.append(
        f"**Sample (10 art./clasă):** lungime medie prop. min: "
        f"cls0 = {s3['cls0_lungime_medie_propmin']:.1f}w, "
        f"cls1 = {s3['cls1_lungime_medie_propmin']:.1f}w. "
        f"Scor mediu min: cls0 = {s3['cls0_scor_mediu_min']:.3f}, "
        f"cls1 = {s3['cls1_scor_mediu_min']:.3f}."
    )
    linii.append("")

    for lbl_key in ("cls0", "cls1"):
        linii.append(f"### Exemple {lbl_key.upper()}")
        linii.append("")
        for i, ex in enumerate(p3["exemple"][lbl_key], 1):
            linii.append(
                f"**{i}.** `{ex['articol_id']}` · {ex['sursa']} · "
                f"{ex['nr_prop_articol']} prop/art · "
                f"prop. min: {ex['nr_cuvinte']}w, scor={ex['scor_min']:.3f}"
            )
            linii.append("")
            linii.append(f"- Prop. min: *{ex['propozitia_min']}*")
            linii.append(f"- Match corpus cls1 (scor={ex['match_scor']:.3f}): "
                         f"*{ex['match_corpus_cls1']}*")
            linii.append("")

    # Proba 4
    linii.append("## Proba 4 — Agregare top-k mean (k=5)")
    linii.append("")
    p4 = raport["proba_4_topk_mean"]
    linii.append(
        f"Recalculare cu **top-{p4['top_k']} mean** în loc de max per propoziție. "
        f"Top-k mean e robust la extreme: cere consistență peste 5 potriviri, "
        f"nu doar o potrivire accidentală."
    )
    linii.append("")
    for test_name, rez in p4["rezultate"].items():
        linii.append(f"### {test_name}")
        linii.append("")
        linii.append("| Agregare | AUC | μ(cls0) | μ(cls1) |")
        linii.append("|---|---:|---:|---:|")
        for agr, m in rez.items():
            linii.append(
                f"| {agr} | {m['auc']:.4f} | {m['media_cls0']:.4f} | "
                f"{m['media_cls1']:.4f} |"
            )
        linii.append("")

    # Concluzii
    linii.append("## Concluzii globale")
    linii.append("")
    for linie in raport["concluzii"]:
        linii.append(f"- {linie}")
    linii.append("")
    linii.append("---")
    linii.append("")
    linii.append("*Modul 3 · Pasul A2.5 · Diagnostic benchmark v4*")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(linii))


if __name__ == "__main__":
    main()
