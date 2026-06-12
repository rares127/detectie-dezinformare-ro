"""
Benchmark embeddings pentru Modulul 3 — Analiza granulara per propozitie.

Scop:
    Compara 3 modele de embeddings pe sarcina de separare cls0 (stiri
    credibile) vs cls1 (dezinformare pro-rusa), pentru a alege modelul
    optim care va fi folosit in pipeline-ul granular complet.

Modele evaluate:
    1. paraphrase-multilingual-MiniLM-L12-v2   (mic, rapid)
    2. paraphrase-multilingual-mpnet-base-v2   (mediu, mai puternic)
    3. XLM-R mean-pooled din checkpoint-ul modulului 2 (fine-tuned pe cls)
       — testeaza daca reprezentarile fine-tuned pe clasificare raman
         utile pentru similaritate semantica.

Combinatii scor evaluate (6 totale):
    - per propozitie: {max, top-5-mean} vs corpus de referinta
    - per articol: {mean, min, p10} agregate pe propozitiile articolului

Metrici raportate:
    - Cohen's d intre distributiile scor_articol pe cls0 vs cls1
    - AUC simplu (Mann-Whitney U normalizat)
    - Separabilitate per sursa (Veridica vs Stopfals pe cls1)
    - Viteza embeddings (propozitii/secunda pe MPS)

Input:
    - data/processed/propozitii_cls0_corpus.parquet (5,290 propozitii)
    - data/processed/subset_benchmark.parquet       (20 articole segmentate)
    - models/xlmr_baseline_v2/final/                (checkpoint XLM-R)

Output:
    - findings/benchmark_embeddings.md
    - findings/benchmark_embeddings.json

Device: MPS cu fallback CPU. Seed 42.

Rulare:
    python scripts/benchmark_embeddings.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


# -----------------------------------------------------------------------------
# Configuratie cablata
# -----------------------------------------------------------------------------
CALE_CORPUS = Path("data/processed/propozitii_cls0_corpus.parquet")
CALE_SUBSET = Path("data/processed/subset_benchmark.parquet")
CALE_CHECKPOINT_XLMR = Path("models/xlmr_baseline_v2/final")

CALE_OUT_MD = Path("findings/benchmark_embeddings.md")
CALE_OUT_JSON = Path("findings/benchmark_embeddings.json")

SEED = 42
BATCH_SIZE = 32
TOP_K = 5
PERCENTILA = 10  # pentru agregarea p10 la nivel de articol

# Modele ST (sentence-transformers). Nume HF.
MODELE_ST = [
    ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("mpnet",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
]

# Numele afisat pentru checkpoint-ul XLM-R fine-tuned
NUME_XLMR = "xlmr_ft_mean"


# -----------------------------------------------------------------------------
# Device + seed
# -----------------------------------------------------------------------------
def alege_device() -> torch.device:
    """Alege MPS daca e disponibil, altfel CPU. CUDA nu e in scope pe M2 Pro."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = SEED) -> None:
    """Seed reproductibil pe toate sursele de randomness."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Embedding engines
# -----------------------------------------------------------------------------
@dataclass
class RezultatEmbed:
    """Containere rezultate embeddings."""
    nume_model: str
    emb_corpus: np.ndarray     # (N_corpus, D), L2-normalized
    emb_articole: np.ndarray   # (N_articole, D), L2-normalized
    viteza_prop_sec: float     # propozitii embedate per secunda


def embed_sentence_transformer(nume_scurt: str, nume_hf: str,
                               propozitii_corpus: list[str],
                               propozitii_articole: list[str],
                               device: torch.device) -> RezultatEmbed:
    """Embed cu sentence-transformers. Normalizare L2 incorporata."""
    print(f"\n[{nume_scurt}] Încarc {nume_hf}...")
    model = SentenceTransformer(nume_hf, device=str(device))
    model.eval()

    # masuram viteza doar pe corpus (setul mare, mai relevant)
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
    """Embed cu checkpoint-ul XLM-R fine-tuned, mean-pooling pe last_hidden_state.

    Atentie: modelul e fine-tuned pentru clasificare, dar extragem
    reprezentarile din encoder (fara head-ul de clasificare).
    """
    print(f"\n[{NUME_XLMR}] Încarc checkpoint din {cale_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(str(cale_checkpoint))
    # AutoModel returneaza encoder-ul fara head-ul de clasificare
    model = AutoModel.from_pretrained(str(cale_checkpoint)).to(device)
    model.eval()

    def encode(texts: list[str], masoara_timp: bool = False) -> tuple[np.ndarray, float]:
        """Encode cu mean-pooling mascat si normalizare L2."""
        toate = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(device)
                out = model(**enc)
                hidden = out.last_hidden_state  # (B, T, D)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                sumat = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                emb = sumat / denom  # mean-pooling mascat
                emb = F.normalize(emb, p=2, dim=1)  # normalizare L2
                toate.append(emb.cpu().numpy())
        elapsed = time.time() - t0
        return np.concatenate(toate, axis=0).astype(np.float32), elapsed

    emb_corpus, elapsed = encode(propozitii_corpus, masoara_timp=True)
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
    """Calculeaza scor per propozitie: max si top-5-mean din similaritatile vs corpus.

    Cosine = dot product pe embeddings normalizate.
    Returneaza dict cu doua array-uri de lungime N_articol_prop.
    """
    sim = emb_art @ emb_corp.T  # (N_art, N_corp)
    scor_max = sim.max(axis=1)
    # top-k mean: sortam descendent pe rand si luam media primelor k
    # np.partition e mai rapid decat sort complet
    k = min(TOP_K, sim.shape[1])
    top_k_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    top_k_vals = np.take_along_axis(sim, top_k_idx, axis=1)
    scor_topk_mean = top_k_vals.mean(axis=1)
    return {"max": scor_max, "topk_mean": scor_topk_mean}


def agrega_pe_articol(df_prop: pd.DataFrame, scoruri_prop: np.ndarray) -> pd.DataFrame:
    """Agrega scorurile per propozitie la nivel de articol: mean, min, p10."""
    df_prop = df_prop.copy()
    df_prop["_scor"] = scoruri_prop
    grp = df_prop.groupby(["articol_id", "sursa_site", "label_numeric"])["_scor"]
    out = pd.DataFrame({
        "mean":  grp.mean(),
        "min":   grp.min(),
        f"p{PERCENTILA}": grp.apply(lambda s: np.percentile(s, PERCENTILA)),
    }).reset_index()
    return out


# -----------------------------------------------------------------------------
# Metrici de separabilitate
# -----------------------------------------------------------------------------
def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d clasic (pooled std). Pozitiv ⇒ a > b in medie."""
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
    """AUC-ROC via Mann-Whitney U. Conventie: scor MARE ⇒ cls0 (credibil).

    Deci `pos` sunt scorurile cls0 (asteptam mai mari, articolele credibile
    sunt similare cu corpusul de referinta cls0), `neg` sunt scorurile cls1.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u / (len(pos) * len(neg)))


def evalueaza_separabilitate(df_scor_art: pd.DataFrame,
                              col_scor: str) -> dict:
    """Calculeaza metrici de separabilitate pe o coloana de scor articol.

    Conventia: scor mai mare = mai similar cu corpus cls0 = mai probabil credibil.
    Deci cls0 ar trebui sa aiba scoruri MAI MARI decat cls1.
    """
    cls0 = df_scor_art[df_scor_art["label_numeric"] == 0][col_scor].values
    cls1 = df_scor_art[df_scor_art["label_numeric"] == 1][col_scor].values

    d = cohen_d(cls0, cls1)
    auc = auc_mannwhitney(cls0, cls1)

    # breakdown per sursa pentru cls1 (Veridica vs Stopfals — critic pentru LOSO)
    veridica = df_scor_art[(df_scor_art["label_numeric"] == 1)
                           & (df_scor_art["sursa_site"] == "veridica.ro")][col_scor].values
    stopfals = df_scor_art[(df_scor_art["label_numeric"] == 1)
                           & (df_scor_art["sursa_site"] == "stopfals.md")][col_scor].values

    return {
        "cls0_mean": float(cls0.mean()) if len(cls0) else float("nan"),
        "cls0_std": float(cls0.std(ddof=1)) if len(cls0) > 1 else float("nan"),
        "cls1_mean": float(cls1.mean()) if len(cls1) else float("nan"),
        "cls1_std": float(cls1.std(ddof=1)) if len(cls1) > 1 else float("nan"),
        "cohen_d": d,
        "auc": auc,
        "veridica_mean": float(veridica.mean()) if len(veridica) else float("nan"),
        "stopfals_mean": float(stopfals.mean()) if len(stopfals) else float("nan"),
        "n_cls0": int(len(cls0)),
        "n_cls1": int(len(cls1)),
    }


# -----------------------------------------------------------------------------
# Raportare markdown
# -----------------------------------------------------------------------------
def genereaza_raport_md(rezultate: dict) -> str:
    """Construieste raportul markdown complet."""
    linii = [
        "# Benchmark embeddings — Modulul 3",
        "",
        f"**Seed:** {SEED} · **Device:** {rezultate['meta']['device']} · "
        f"**Top-K:** {TOP_K} · **Percentilă:** p{PERCENTILA}",
        "",
        f"**Corpus referință:** {rezultate['meta']['n_corpus']} propoziții",
        f"**Subset benchmark:** {rezultate['meta']['n_articole']} articole, "
        f"{rezultate['meta']['n_propozitii']} propoziții",
        "",
        "## Tabel principal — separabilitate cls0 vs cls1",
        "",
        "Convenție: scor mai mare = mai similar cu corpus cls0 = mai credibil.",
        "Deci cls0 ar trebui să aibă scoruri mai mari decât cls1.",
        "",
        "AUC > 0.5 ⇒ direcție corectă. Cohen's d > 0 ⇒ direcție corectă.",
        "",
        "| Model | Prop-scor | Art-agregare | AUC | Cohen's d | μ(cls0) | μ(cls1) | μ(Veridica) | μ(Stopfals) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    # sortam dupa AUC descendent ca sa vedem primul top
    randuri = []
    for model_nume, rez_model in rezultate["modele"].items():
        for prop_scor in ["max", "topk_mean"]:
            for art_ag in ["mean", "min", f"p{PERCENTILA}"]:
                m = rez_model["scoruri"][prop_scor][art_ag]
                randuri.append({
                    "model": model_nume,
                    "prop_scor": prop_scor,
                    "art_ag": art_ag,
                    "auc": m["auc"],
                    "d": m["cohen_d"],
                    "c0": m["cls0_mean"],
                    "c1": m["cls1_mean"],
                    "ver": m["veridica_mean"],
                    "sto": m["stopfals_mean"],
                })
    randuri.sort(key=lambda r: (-(r["auc"] if not np.isnan(r["auc"]) else -1)))

    for r in randuri:
        linii.append(
            f"| {r['model']} | {r['prop_scor']} | {r['art_ag']} | "
            f"{r['auc']:.3f} | {r['d']:+.2f} | {r['c0']:.3f} | {r['c1']:.3f} | "
            f"{r['ver']:.3f} | {r['sto']:.3f} |"
        )

    linii += [
        "",
        "## Viteză embeddings (corpus cls0, 5,290 propoziții)",
        "",
        "| Model | Propoziții/secundă | Timp total corpus (s) |",
        "|---|---|---|",
    ]
    for model_nume, rez_model in rezultate["modele"].items():
        v = rez_model["viteza_prop_sec"]
        t = rezultate["meta"]["n_corpus"] / v if v > 0 else float("nan")
        linii.append(f"| {model_nume} | {v:.1f} | {t:.1f} |")

    linii += [
        "",
        "## Interpretare rapidă",
        "",
        f"**Top configurație după AUC:** `{randuri[0]['model']}` cu "
        f"propoziție={randuri[0]['prop_scor']}, articol={randuri[0]['art_ag']} "
        f"(AUC={randuri[0]['auc']:.3f}, d={randuri[0]['d']:+.2f}).",
        "",
        "### Praguri de decizie orientative",
        "- AUC ≥ 0.90: separabilitate foarte bună → scor granular poate fi semnal puternic",
        "- AUC 0.75–0.90: separabilitate bună → util în combinație cu clasificatorul global",
        "- AUC 0.60–0.75: semnal slab → util doar ca feature auxiliar",
        "- AUC < 0.60: aproape aleator → re-gândim abordarea",
        "",
        "### Cross-source (relevant pentru problema LOSO-V)",
        "Compară μ(Veridica) vs μ(Stopfals) pentru configurația câștigătoare.",
        "Dacă diferența e mică (≤0.02), modelul tratează ambele surse similar ⇒",
        "semnal cross-source robust. Dacă diferența e mare, avem încă stylistic fingerprint.",
        "",
        "*Generat automat.*",
    ]
    return "\n".join(linii)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    """Ruleaza benchmark-ul complet."""
    print("=" * 70)
    print("BENCHMARK EMBEDDINGS — MODULUL 3")
    print("=" * 70)

    seed_all()
    device = alege_device()
    print(f"Device: {device}")

    # 1. incarcare date
    if not CALE_CORPUS.exists():
        raise FileNotFoundError(f"Nu găsesc corpus-ul la {CALE_CORPUS}")
    if not CALE_SUBSET.exists():
        raise FileNotFoundError(
            f"Nu găsesc subset-ul la {CALE_SUBSET}. "
            f"Rulează mai întâi selecteaza_subset_benchmark.py."
        )

    df_corp = pd.read_parquet(CALE_CORPUS)
    df_sub = pd.read_parquet(CALE_SUBSET)
    print(f"\nCorpus: {len(df_corp)} propoziții")
    print(f"Subset: {df_sub['articol_id'].nunique()} articole, "
          f"{len(df_sub)} propoziții")

    propozitii_corpus = df_corp["propozitie"].tolist()
    propozitii_articole = df_sub["propozitie"].tolist()

    # 2. embed cu fiecare model
    rezultate_embed = {}

    for nume_scurt, nume_hf in MODELE_ST:
        rez = embed_sentence_transformer(
            nume_scurt, nume_hf, propozitii_corpus, propozitii_articole, device
        )
        rezultate_embed[nume_scurt] = rez

    # XLM-R doar daca checkpoint-ul exista
    if CALE_CHECKPOINT_XLMR.exists():
        rez = embed_xlmr_mean_pooled(
            CALE_CHECKPOINT_XLMR, propozitii_corpus, propozitii_articole, device
        )
        rezultate_embed[NUME_XLMR] = rez
    else:
        print(f"\n⚠️  Checkpoint XLM-R nu a fost găsit la {CALE_CHECKPOINT_XLMR}. "
              f"Sar peste acest model.")

    # 3. calculeaza scoruri si metrici
    output = {
        "meta": {
            "seed": SEED,
            "device": str(device),
            "n_corpus": len(df_corp),
            "n_articole": int(df_sub["articol_id"].nunique()),
            "n_propozitii": len(df_sub),
            "top_k": TOP_K,
            "percentila": PERCENTILA,
        },
        "modele": {},
    }

    for nume_model, rez in rezultate_embed.items():
        print(f"\n--- Metrici pentru {nume_model} ---")
        scoruri = scoruri_propozitie(rez.emb_articole, rez.emb_corpus)
        metrici_model = {}
        for prop_key, scor_prop in scoruri.items():
            df_art = agrega_pe_articol(df_sub, scor_prop)
            metrici_model[prop_key] = {}
            for ag_col in ["mean", "min", f"p{PERCENTILA}"]:
                m = evalueaza_separabilitate(df_art, ag_col)
                metrici_model[prop_key][ag_col] = m
                print(f"  {prop_key}/{ag_col}: AUC={m['auc']:.3f}, "
                      f"d={m['cohen_d']:+.2f}, "
                      f"μ0={m['cls0_mean']:.3f} vs μ1={m['cls1_mean']:.3f}")

        output["modele"][nume_model] = {
            "viteza_prop_sec": rez.viteza_prop_sec,
            "scoruri": metrici_model,
        }

    # 4. salvare outputs
    CALE_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    CALE_OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"\n✅ JSON: {CALE_OUT_JSON}")

    raport = genereaza_raport_md(output)
    CALE_OUT_MD.write_text(raport, encoding="utf-8")
    print(f"✅ Raport MD: {CALE_OUT_MD}")


if __name__ == "__main__":
    main()
