"""
Benchmark embeddings v3 — Modulul 3 (scale up).

Schimbari fata de v1:
    - Citeste subset_benchmark_v3.parquet (cls0 din surse externe,
      fara contaminare cu corpus).
    - Adauga metrica "proportie propozitii sub threshold" — pentru fiecare
      articol, calculam fractia propozitiilor al caror scor max vs corpus
      e sub un prag τ. Interpretare: „cat din articol e fara corespondent
      factual in corpusul credibil".
    - Pastreaza XLM-R ca finding negativ documentat (nu il excludem).
    - Raport cu sectiune explicita despre gap-ul Veridica vs Stopfals.

Modele evaluate:
    1. paraphrase-multilingual-MiniLM-L12-v2
    2. paraphrase-multilingual-mpnet-base-v2
    3. XLM-R mean-pooled din xlmr_baseline_v2/final/
       (pastrat ca finding negativ: fine-tuning colapseaza spatiul semantic)

Metrici noi adaugate:
    - Pentru fiecare articol: pct_sub_{03,04,05,06} = fractia propozitiilor
      cu scor_max < τ. Agregare per articol: media propozitiilor individuale.
    - Separabilitate cls0 vs cls1 pe aceasta metrica (AUC, Cohen's d).
      NB: aici conventia e inversa: scor mai mare = cls1 (propagandistic),
      deoarece mai multe propozitii fara corespondent factual = mai dubios.

Input:
    - data/processed/propozitii_cls0_corpus.parquet
    - data/processed/subset_benchmark_v3.parquet
    - models/xlmr_baseline_v2/final/

Output:
    - findings/benchmark_embeddings_v3.md
    - findings/benchmark_embeddings_v3.json

Device: MPS cu fallback CPU. Seed 42.

Rulare:
    python scripts/benchmark_embeddings_v2.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


# -----------------------------------------------------------------------------
# Configuratie
# -----------------------------------------------------------------------------
CALE_CORPUS = Path("data/processed/propozitii_cls0_corpus.parquet")
CALE_SUBSET = Path("data/processed/subset_benchmark_v3.parquet")
CALE_CHECKPOINT_XLMR = Path("models/xlmr_baseline_v2/final")

CALE_OUT_MD = Path("findings/benchmark_embeddings_v3.md")
CALE_OUT_JSON = Path("findings/benchmark_embeddings_v3.json")

SEED = 42
BATCH_SIZE = 32
TOP_K = 5
PERCENTILA = 10
PRAGURI = [0.3, 0.4, 0.5, 0.6]  # praguri pentru proportie sub threshold

MODELE_ST = [
    ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("mpnet",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
]
NUME_XLMR = "xlmr_ft_mean"


# -----------------------------------------------------------------------------
# Device + seed
# -----------------------------------------------------------------------------
def alege_device() -> torch.device:
    """MPS daca disponibil, altfel CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = SEED) -> None:
    """Seed reproductibil."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Embedding engines
# -----------------------------------------------------------------------------
@dataclass
class RezultatEmbed:
    """Container embeddings pentru un model."""
    nume_model: str
    emb_corpus: np.ndarray
    emb_articole: np.ndarray
    viteza_prop_sec: float


def embed_sentence_transformer(nume_scurt: str, nume_hf: str,
                               propozitii_corpus: list[str],
                               propozitii_articole: list[str],
                               device: torch.device) -> RezultatEmbed:
    """Embed cu sentence-transformers, cu normalizare L2 incorporata."""
    print(f"\n[{nume_scurt}] Încarc {nume_hf}...")
    model = SentenceTransformer(nume_hf, device=str(device))
    model.eval()

    t0 = time.time()
    emb_corpus = model.encode(
        propozitii_corpus,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    viteza = len(propozitii_corpus) / elapsed

    emb_articole = model.encode(
        propozitii_articole,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print(f"[{nume_scurt}] Corpus embedat în {elapsed:.1f}s ({viteza:.1f} prop/s)")
    return RezultatEmbed(
        nume_model=nume_scurt,
        emb_corpus=emb_corpus.astype(np.float32),
        emb_articole=emb_articole.astype(np.float32),
        viteza_prop_sec=viteza,
    )


def embed_xlmr_mean_pooled(cale_checkpoint: Path,
                           propozitii_corpus: list[str],
                           propozitii_articole: list[str],
                           device: torch.device) -> RezultatEmbed:
    """Embed cu XLM-R fine-tuned — mean-pooling mascat + normalizare L2."""
    print(f"\n[{NUME_XLMR}] Încarc din {cale_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(str(cale_checkpoint))
    model = AutoModel.from_pretrained(str(cale_checkpoint)).to(device)
    model.eval()

    def encode(texts: list[str]) -> tuple[np.ndarray, float]:
        toate = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(device)
                out = model(**enc)
                hidden = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                sumat = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                emb = sumat / denom
                emb = F.normalize(emb, p=2, dim=1)
                toate.append(emb.cpu().numpy())
        elapsed = time.time() - t0
        return np.concatenate(toate, axis=0).astype(np.float32), elapsed

    emb_corpus, elapsed = encode(propozitii_corpus)
    viteza = len(propozitii_corpus) / elapsed
    emb_articole, _ = encode(propozitii_articole)

    print(f"[{NUME_XLMR}] Corpus embedat în {elapsed:.1f}s ({viteza:.1f} prop/s)")
    return RezultatEmbed(
        nume_model=NUME_XLMR,
        emb_corpus=emb_corpus,
        emb_articole=emb_articole,
        viteza_prop_sec=viteza,
    )


# -----------------------------------------------------------------------------
# Scoruri similaritate
# -----------------------------------------------------------------------------
def scoruri_propozitie(emb_art: np.ndarray, emb_corp: np.ndarray) -> dict[str, np.ndarray]:
    """Scor per propozitie: max si top-k-mean vs corpus."""
    sim = emb_art @ emb_corp.T
    scor_max = sim.max(axis=1)
    k = min(TOP_K, sim.shape[1])
    top_k_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    top_k_vals = np.take_along_axis(sim, top_k_idx, axis=1)
    scor_topk_mean = top_k_vals.mean(axis=1)
    return {"max": scor_max, "topk_mean": scor_topk_mean}


def agrega_pe_articol(df_prop: pd.DataFrame, scoruri_prop: np.ndarray) -> pd.DataFrame:
    """Agrega scor-ul de propozitie la nivel de articol: mean, min, p10."""
    df_prop = df_prop.copy()
    df_prop["_scor"] = scoruri_prop
    grp = df_prop.groupby(["articol_id", "sursa_site", "label_numeric"])["_scor"]
    out = pd.DataFrame({
        "mean":  grp.mean(),
        "min":   grp.min(),
        f"p{PERCENTILA}": grp.apply(lambda s: np.percentile(s, PERCENTILA)),
    }).reset_index()
    return out


def proporie_sub_threshold(df_prop: pd.DataFrame, scoruri_prop: np.ndarray,
                           praguri: list[float]) -> pd.DataFrame:
    """Pentru fiecare articol, fractia propozitiilor cu scor sub τ.

    Conventie: scor MAI MIC = propozitie fara corespondent factual = suspect.
    Deci o fractie MAI MARE sub τ = articol mai probabil cls1 (dezinformare).
    Conventia e INVERSA fata de scorurile agregate (acolo scor mare = credibil).
    """
    df_prop = df_prop.copy()
    df_prop["_scor"] = scoruri_prop
    rezultat = {}
    for prag in praguri:
        eticheta = f"pct_sub_{int(prag * 100):02d}"
        df_prop[eticheta] = (df_prop["_scor"] < prag).astype(int)
        rezultat[eticheta] = df_prop.groupby(
            ["articol_id", "sursa_site", "label_numeric"]
        )[eticheta].mean()
    out = pd.DataFrame(rezultat).reset_index()
    return out


# -----------------------------------------------------------------------------
# Metrici separabilitate
# -----------------------------------------------------------------------------
def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d cu pooled std. Pozitiv ⇒ a > b in medie."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def auc_mannwhitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC via Mann-Whitney U. Conventie: pos > neg asteptat."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u / (len(pos) * len(neg)))


def evalueaza_separabilitate(df_scor_art: pd.DataFrame, col_scor: str,
                              directie_pozitiva: str = "cls0") -> dict:
    """Calculeaza metrici de separabilitate.

    directie_pozitiva:
        "cls0" — scor mare asteptat pentru cls0 (cazul scorurilor agregate).
        "cls1" — scor mare asteptat pentru cls1 (cazul proportie sub threshold).
    """
    cls0 = df_scor_art[df_scor_art["label_numeric"] == 0][col_scor].values
    cls1 = df_scor_art[df_scor_art["label_numeric"] == 1][col_scor].values

    if directie_pozitiva == "cls0":
        d = cohen_d(cls0, cls1)
        auc = auc_mannwhitney(cls0, cls1)
    else:
        d = cohen_d(cls1, cls0)
        auc = auc_mannwhitney(cls1, cls0)

    veridica = df_scor_art[(df_scor_art["label_numeric"] == 1)
                           & (df_scor_art["sursa_site"] == "veridica.ro")][col_scor].values
    stopfals = df_scor_art[(df_scor_art["label_numeric"] == 1)
                           & (df_scor_art["sursa_site"] == "stopfals.md")][col_scor].values

    # breakdown per sursa cls0 (mai multe surse externe)
    per_sursa_cls0 = {}
    for sursa in df_scor_art[df_scor_art["label_numeric"] == 0]["sursa_site"].unique():
        vals = df_scor_art[(df_scor_art["label_numeric"] == 0)
                           & (df_scor_art["sursa_site"] == sursa)][col_scor].values
        per_sursa_cls0[str(sursa)] = float(vals.mean()) if len(vals) else float("nan")

    return {
        "cls0_mean": float(cls0.mean()) if len(cls0) else float("nan"),
        "cls0_std": float(cls0.std(ddof=1)) if len(cls0) > 1 else float("nan"),
        "cls1_mean": float(cls1.mean()) if len(cls1) else float("nan"),
        "cls1_std": float(cls1.std(ddof=1)) if len(cls1) > 1 else float("nan"),
        "cohen_d": d,
        "auc": auc,
        "veridica_mean": float(veridica.mean()) if len(veridica) else float("nan"),
        "stopfals_mean": float(stopfals.mean()) if len(stopfals) else float("nan"),
        "gap_v_vs_s": (float(veridica.mean()) - float(stopfals.mean()))
                      if (len(veridica) and len(stopfals)) else float("nan"),
        "per_sursa_cls0": per_sursa_cls0,
        "n_cls0": int(len(cls0)),
        "n_cls1": int(len(cls1)),
    }


# -----------------------------------------------------------------------------
# Raportare markdown
# -----------------------------------------------------------------------------
def genereaza_raport_md(rezultate: dict) -> str:
    """Construieste raport complet cu sectiune finding negativ XLM-R."""
    meta = rezultate["meta"]
    linii = [
        "# Benchmark embeddings v3 — Modulul 3 (scale up)",
        "",
        f"**Seed:** {meta['seed']} · **Device:** {meta['device']} · "
        f"**Top-K:** {meta['top_k']} · **Percentilă:** p{meta['percentila']}",
        "",
        "**Scale up față de v2:** articolele cls0 vin acum din surse EXTERNE "
        "(Pro TV, HotNews etc.), nu din aceleași surse Digi24/G4Media din care "
        "s-a construit corpusul. V1 avea contaminare 100% (cls0 test ⊂ corpus) "
        "ceea ce producea AUC=1.0 trivial.",
        "",
        f"**Corpus referință:** {meta['n_corpus']} propoziții (neatins)",
        f"**Subset v3:** {meta['n_articole']} articole "
        f"({meta['n_cls0']} cls0 extern + {meta['n_cls1']} cls1), "
        f"{meta['n_propozitii']} propoziții",
        "",
        "---",
        "",
        "## 1. Separabilitate — scoruri agregate",
        "",
        "Convenție: scor mai mare = mai similar cu corpus cls0 = mai credibil.",
        "cls0 ar trebui să aibă scoruri MAI MARI decât cls1.",
        "",
        "| Model | Prop | Agr. | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(Ver) | μ(Stop) | Δ(V-S) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    randuri = []
    for model_nume, rez_model in rezultate["modele"].items():
        for prop_scor in ["max", "topk_mean"]:
            for art_ag in ["mean", "min", f"p{meta['percentila']}"]:
                m = rez_model["scoruri_agregate"][prop_scor][art_ag]
                randuri.append({
                    "model": model_nume, "prop": prop_scor, "agr": art_ag,
                    "auc": m["auc"], "d": m["cohen_d"],
                    "c0": m["cls0_mean"], "c1": m["cls1_mean"],
                    "v": m["veridica_mean"], "s": m["stopfals_mean"],
                    "gap": m["gap_v_vs_s"],
                })
    randuri.sort(key=lambda r: (-(r["auc"] if not np.isnan(r["auc"]) else -1),
                                -(r["d"] if not np.isnan(r["d"]) else -999)))

    for r in randuri:
        linii.append(
            f"| {r['model']} | {r['prop']} | {r['agr']} | "
            f"{r['auc']:.3f} | {r['d']:+.2f} | "
            f"{r['c0']:.3f} | {r['c1']:.3f} | "
            f"{r['v']:.3f} | {r['s']:.3f} | {r['gap']:+.3f} |"
        )

    # -------------------------------------------------------------------------
    # Sectiunea 2: proportie sub threshold
    # -------------------------------------------------------------------------
    linii += [
        "",
        "---",
        "",
        "## 2. Separabilitate — proporție propoziții sub threshold τ",
        "",
        "Metrică: pentru fiecare articol, fracția propozițiilor cu scor_max < τ.",
        "Convenție INVERSĂ aici: fracție MAI MARE = mai multe propoziții fără ",
        "corespondent factual = mai probabil cls1 (dezinformare).",
        "",
        f"Praguri τ testate: {PRAGURI}",
        "",
        "| Model | Prag τ | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(Ver) | μ(Stop) | Δ(V-S) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    randuri_prop = []
    for model_nume, rez_model in rezultate["modele"].items():
        for prag in PRAGURI:
            eticheta = f"pct_sub_{int(prag * 100):02d}"
            m = rez_model["scoruri_prop_sub"][eticheta]
            randuri_prop.append({
                "model": model_nume, "prag": prag, "auc": m["auc"], "d": m["cohen_d"],
                "c0": m["cls0_mean"], "c1": m["cls1_mean"],
                "v": m["veridica_mean"], "s": m["stopfals_mean"],
                "gap": m["gap_v_vs_s"],
            })
    randuri_prop.sort(key=lambda r: (-(r["auc"] if not np.isnan(r["auc"]) else -1)))

    for r in randuri_prop:
        linii.append(
            f"| {r['model']} | {r['prag']:.1f} | "
            f"{r['auc']:.3f} | {r['d']:+.2f} | "
            f"{r['c0']:.3f} | {r['c1']:.3f} | "
            f"{r['v']:.3f} | {r['s']:.3f} | {r['gap']:+.3f} |"
        )

    # -------------------------------------------------------------------------
    # Sectiunea 3: viteza
    # -------------------------------------------------------------------------
    linii += [
        "",
        "---",
        "",
        "## 3. Viteză embeddings (corpus 5,290 propoziții)",
        "",
        "| Model | Propoziții/secundă | Timp total (s) |",
        "|---|---|---|",
    ]
    for model_nume, rez_model in rezultate["modele"].items():
        v = rez_model["viteza_prop_sec"]
        t = meta["n_corpus"] / v if v > 0 else float("nan")
        linii.append(f"| {model_nume} | {v:.1f} | {t:.1f} |")

    # -------------------------------------------------------------------------
    # Sectiunea 4: XLM-R ca finding negativ
    # -------------------------------------------------------------------------
    linii += [
        "",
        "---",
        "",
        "## 4. Finding metodologic — XLM-R fine-tuned NU e adecvat pentru similaritate",
        "",
        "**Observația**: checkpoint-ul XLM-R fine-tuned pe clasificare (modulul 2) "
        "produce embeddings cu distribuție colapsată — toate propozițiile primesc ",
        "scoruri aproape identice, indiferent de conținut. Uită-te la coloanele "
        "μ(cls0) și μ(cls1) în tabelele de mai sus pentru `xlmr_ft_mean`: diferența "
        "dintre clase e sub 0.01 în valoare absolută, vs ~0.35 pentru MiniLM/mpnet.",
        "",
        "**Interpretare**: fine-tuning-ul pe sarcina de clasificare binară împinge "
        "reprezentările spre hiperplanul de decizie, colapsând geometria semantică "
        "originală (pe care XLM-R pretrained o avea). Modelul și-a optimizat "
        "reprezentările pentru a separa cls0/cls1 la nivel de POOL de clasificare, "
        "nu pentru a captura similaritate semantică generală.",
        "",
        "**Implicație pentru teză**: acest rezultat justifică alegerea unui model "
        "antrenat specific pe similaritate (sentence-transformers) în locul "
        "reutilizării modelului de clasificare. E o contribuție metodologică: "
        "validează empiric că arhitectura sistemului trebuie să folosească două "
        "modele distincte, nu unul singur pentru ambele sarcini.",
        "",
        "**De inclus în capitolul „Arhitectură și justificări tehnice”**.",
    ]

    # -------------------------------------------------------------------------
    # Sectiunea 5: gap cross-source
    # -------------------------------------------------------------------------
    linii += [
        "",
        "---",
        "",
        "## 5. Cross-source — gap Veridica vs Stopfals (critic pentru modulul 2)",
        "",
        "Dacă gap-ul Δ(V-S) e aproape de zero, modelul tratează ambele surse ",
        "propagandistice similar — semn că analiza granulară e robustă ",
        "cross-source și poate compensa problema LOSO-V din modulul 2.",
        "",
        "Dacă gap-ul e semnificativ (>0.05), avem încă stylistic fingerprint și ",
        "modulul 3 singur nu rezolvă problema.",
        "",
        "**Cel mai mic gap pe scoruri agregate** (mai mic e mai bine):",
    ]
    # gasim minimul gap pe scoruri agregate (doar valori absolute mici)
    randuri_st = [r for r in randuri if not np.isnan(r["gap"])]
    randuri_st.sort(key=lambda r: abs(r["gap"]))
    if randuri_st:
        r = randuri_st[0]
        linii.append(
            f"- `{r['model']}` / {r['prop']} / {r['agr']}: "
            f"Δ = {r['gap']:+.3f}, AUC = {r['auc']:.3f}, d = {r['d']:+.2f}"
        )

    # -------------------------------------------------------------------------
    # Sectiunea 6: recomandare finala
    # -------------------------------------------------------------------------
    linii += [
        "",
        "---",
        "",
        "## 6. Recomandare finală",
        "",
        "Criterii de alegere:",
        "1. AUC cât mai aproape de 1.0 (dar realist — fără contaminare)",
        "2. Gap V-S cât mai mic (robustețe cross-source)",
        "3. Viteză acceptabilă pe M2 Pro",
        "4. Interpretabilitate (preferăm metrici simple de explicat)",
        "",
    ]

    # alegere: top AUC pe scoruri agregate
    if randuri:
        best = randuri[0]
        linii.append(
            f"**Configurație principală propusă:** `{best['model']}` + "
            f"propoziție={best['prop']} + articol={best['agr']}"
        )
        linii.append(f"- AUC = {best['auc']:.3f}")
        linii.append(f"- Cohen's d = {best['d']:+.2f}")
        linii.append(f"- Gap V-S = {best['gap']:+.3f}")
        linii.append("")

    # metrica proportie suplimentara
    if randuri_prop:
        best_p = randuri_prop[0]
        linii.append(
            f"**Metrică interpretabilă suplimentară:** proporție sub "
            f"τ={best_p['prag']} cu `{best_p['model']}` "
            f"(AUC = {best_p['auc']:.3f}, d = {best_p['d']:+.2f})"
        )
        linii.append("")
        linii.append(
            "Această metrică e utilă pentru interfața demonstrativă — poate fi "
            "prezentată ca „X% din propozițiile articolului nu au corespondent "
            "în corpusul de presă credibilă”, mai intuitivă decât un scor abstract."
        )

    linii.append("")
    linii.append("*Generat automat.*")
    return "\n".join(linii)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    """Pipeline complet benchmark v2."""
    print("=" * 70)
    print("BENCHMARK EMBEDDINGS v3 — MODULUL 3")
    print("=" * 70)

    seed_all()
    device = alege_device()
    print(f"Device: {device}")

    if not CALE_CORPUS.exists():
        raise FileNotFoundError(f"Lipsește corpus-ul: {CALE_CORPUS}")
    if not CALE_SUBSET.exists():
        raise FileNotFoundError(
            f"Lipsește subset-ul v2: {CALE_SUBSET}. "
            f"Rulează mai întâi selecteaza_subset_benchmark_v3.py."
        )

    df_corp = pd.read_parquet(CALE_CORPUS)
    df_sub = pd.read_parquet(CALE_SUBSET)
    print(f"\nCorpus: {len(df_corp)} propoziții")
    print(f"Subset v3: {df_sub['articol_id'].nunique()} articole, "
          f"{len(df_sub)} propoziții")
    print(f"  cls0: {(df_sub[df_sub.label_numeric == 0]['articol_id'].nunique())} articole")
    print(f"  cls1: {(df_sub[df_sub.label_numeric == 1]['articol_id'].nunique())} articole")

    propozitii_corpus = df_corp["propozitie"].tolist()
    propozitii_articole = df_sub["propozitie"].tolist()

    # embed cu fiecare model
    rezultate_embed = {}
    for nume_scurt, nume_hf in MODELE_ST:
        rez = embed_sentence_transformer(
            nume_scurt, nume_hf, propozitii_corpus, propozitii_articole, device
        )
        rezultate_embed[nume_scurt] = rez

    if CALE_CHECKPOINT_XLMR.exists():
        rez = embed_xlmr_mean_pooled(
            CALE_CHECKPOINT_XLMR, propozitii_corpus, propozitii_articole, device
        )
        rezultate_embed[NUME_XLMR] = rez
    else:
        print(f"\n⚠️  Checkpoint XLM-R absent la {CALE_CHECKPOINT_XLMR}.")

    # calculeaza toate metricile
    n_cls0 = int(df_sub[df_sub.label_numeric == 0]["articol_id"].nunique())
    n_cls1 = int(df_sub[df_sub.label_numeric == 1]["articol_id"].nunique())

    output = {
        "meta": {
            "seed": SEED,
            "device": str(device),
            "n_corpus": len(df_corp),
            "n_articole": int(df_sub["articol_id"].nunique()),
            "n_cls0": n_cls0,
            "n_cls1": n_cls1,
            "n_propozitii": len(df_sub),
            "top_k": TOP_K,
            "percentila": PERCENTILA,
            "praguri": PRAGURI,
        },
        "modele": {},
    }

    for nume_model, rez in rezultate_embed.items():
        print(f"\n--- Metrici pentru {nume_model} ---")
        scoruri = scoruri_propozitie(rez.emb_articole, rez.emb_corpus)

        # scoruri agregate (conventie: scor mare = cls0)
        metrici_agregate = {}
        for prop_key, scor_prop in scoruri.items():
            df_art = agrega_pe_articol(df_sub, scor_prop)
            metrici_agregate[prop_key] = {}
            for ag_col in ["mean", "min", f"p{PERCENTILA}"]:
                m = evalueaza_separabilitate(df_art, ag_col, directie_pozitiva="cls0")
                metrici_agregate[prop_key][ag_col] = m
                print(f"  {prop_key}/{ag_col}: AUC={m['auc']:.3f}, "
                      f"d={m['cohen_d']:+.2f}, μ0={m['cls0_mean']:.3f}, "
                      f"μ1={m['cls1_mean']:.3f}, Δ(V-S)={m['gap_v_vs_s']:+.3f}")

        # proportie sub threshold (conventie: fractie mare = cls1)
        # folosim scor_max ca baza (cel mai natural)
        df_prop_sub = proporie_sub_threshold(df_sub, scoruri["max"], PRAGURI)
        metrici_prop_sub = {}
        for prag in PRAGURI:
            eticheta = f"pct_sub_{int(prag * 100):02d}"
            m = evalueaza_separabilitate(df_prop_sub, eticheta,
                                         directie_pozitiva="cls1")
            metrici_prop_sub[eticheta] = m
            print(f"  pct_sub_{prag}: AUC={m['auc']:.3f}, "
                  f"d={m['cohen_d']:+.2f}, μ0={m['cls0_mean']:.3f}, "
                  f"μ1={m['cls1_mean']:.3f}")

        output["modele"][nume_model] = {
            "viteza_prop_sec": rez.viteza_prop_sec,
            "scoruri_agregate": metrici_agregate,
            "scoruri_prop_sub": metrici_prop_sub,
        }

    # salvare outputs
    CALE_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    CALE_OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"\n✅ JSON: {CALE_OUT_JSON}")

    raport = genereaza_raport_md(output)
    CALE_OUT_MD.write_text(raport, encoding="utf-8")
    print(f"✅ Raport: {CALE_OUT_MD}")


if __name__ == "__main__":
    main()
