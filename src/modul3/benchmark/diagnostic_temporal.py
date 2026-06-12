"""
Diagnostic temporal — verifica simetria corpusului si alinierea cu subset-ul v2.

Scop:
    Raspunde la intrebarea „e corpusul nostru simetric temporal?" si
    verifica daca distributia temporala a articolelor externe din
    subset_benchmark_v2 se potriveste cu cea a corpusului.

    Daca NU se potriveste, AUC-ul de 0.733 poate fi subestimat din
    cauza mismatch-ului temporal, nu din cauza unei limitari fundamentale
    a similaritatii semantice.

Rulare:
    python scripts/diagnostic_temporal.py
"""

from pathlib import Path
import pandas as pd

CALE_CORPUS = Path("data/processed/propozitii_cls0_corpus.parquet")
CALE_SUBSET = Path("data/processed/subset_benchmark_v2.parquet")
CALE_CLS0_EXTERN = Path("data/raw/test_cls0_external.csv")
CALE_DATASET_COMPLET = Path("data/raw/dataset_licenta_complet.csv")


def main():
    """Ruleaza toate verificarile si afiseaza in consola."""
    print("=" * 70)
    print("DIAGNOSTIC TEMPORAL CORPUS vs SUBSET v2")
    print("=" * 70)

    # 1. distributie temporala corpus
    print("\n[1] DISTRIBUȚIE TEMPORALĂ CORPUS (propoziții)")
    print("-" * 50)
    df_corp = pd.read_parquet(CALE_CORPUS)
    print(f"Total: {len(df_corp):,} propoziții")
    print("\nPer an:")
    per_an = df_corp["an"].value_counts().sort_index()
    for an, n in per_an.items():
        pct = n / len(df_corp) * 100
        bar = "█" * int(pct / 2)
        print(f"  {an}: {n:>5,} ({pct:>5.1f}%) {bar}")

    print("\nPer an × sursă:")
    cross = pd.crosstab(df_corp["an"], df_corp["sursa_site"], margins=True)
    print(cross.to_string())

    # 2. distributie temporala cls0 extern
    print("\n[2] DISTRIBUȚIE TEMPORALĂ cls0 EXTERN (articole)")
    print("-" * 50)
    df_ext = pd.read_csv(CALE_CLS0_EXTERN)
    print(f"Total: {len(df_ext)} articole")
    print("\nPer an:")
    per_an_ext = df_ext["an"].value_counts().sort_index()
    for an, n in per_an_ext.items():
        pct = n / len(df_ext) * 100
        bar = "█" * int(pct / 3)
        print(f"  {an}: {n:>3} ({pct:>5.1f}%) {bar}")

    # 3. distributie temporala cls1 din dataset complet
    print("\n[3] DISTRIBUȚIE TEMPORALĂ cls1 (Veridica + Stopfals, articole)")
    print("-" * 50)
    df_all = pd.read_csv(CALE_DATASET_COMPLET)
    df_cls1 = df_all[df_all["label_numeric"] == 1]
    print(f"Total: {len(df_cls1)} articole")
    print("\nPer an:")
    per_an_cls1 = df_cls1["an"].value_counts().sort_index()
    for an, n in per_an_cls1.items():
        pct = n / len(df_cls1) * 100
        bar = "█" * int(pct / 2)
        print(f"  {an}: {n:>3} ({pct:>5.1f}%) {bar}")

    # 4. comparatie directa — exista mismatch?
    print("\n[4] ALINIERE TEMPORALĂ — corpus vs cls0 extern")
    print("-" * 50)
    ani_corp = set(df_corp["an"].unique())
    ani_ext = set(df_ext["an"].unique())
    ani_lipsa_corp = ani_ext - ani_corp
    if ani_lipsa_corp:
        print(f"⚠️  Ani în cls0 extern DAR NU în corpus: {ani_lipsa_corp}")
        print("   Propozițiile din acei ani nu au corespondent temporal în corpus.")
    else:
        print("✓ Toți anii din cls0 extern există și în corpus.")

    # procent corp din anii cls0 extern
    prop_corp_in_ani_ext = df_corp[df_corp["an"].isin(ani_ext)]
    pct = len(prop_corp_in_ani_ext) / len(df_corp) * 100
    print(f"\nPropoziții corpus din anii reprezentați în cls0 extern: "
          f"{len(prop_corp_in_ani_ext):,} ({pct:.1f}%)")

    # distributia relativa: mismatch
    print("\nMismatch per an (% corpus vs % cls0 extern):")
    ani_union = sorted(ani_corp | ani_ext)
    for an in ani_union:
        pct_corp = (df_corp["an"] == an).sum() / len(df_corp) * 100
        pct_ext = (df_ext["an"] == an).sum() / len(df_ext) * 100
        diff = pct_ext - pct_corp
        simbol = "⚠️ " if abs(diff) > 15 else "  "
        print(f"  {simbol} {an}: corpus={pct_corp:>5.1f}% | ext={pct_ext:>5.1f}% | "
              f"diff={diff:+.1f}pp")

    # 5. concluzie
    print("\n[5] CONCLUZIE DIAGNOSTIC")
    print("-" * 50)
    mismatch_mare = False
    for an in ani_union:
        pct_corp = (df_corp["an"] == an).sum() / len(df_corp) * 100
        pct_ext = (df_ext["an"] == an).sum() / len(df_ext) * 100
        if abs(pct_ext - pct_corp) > 15:
            mismatch_mare = True
            break

    if mismatch_mare:
        print("⚠️  Există mismatch temporal semnificativ (>15pp pe cel puțin un an).")
        print("   Recomandare: la scraping, țintim distribuție temporală aliniată")
        print("   cu corpusul (nu cu distribuția naturală actuală a surselor).")
    else:
        print("✓ Alinierea temporală e rezonabilă. Scraping-ul poate continua")
        print("  fără constrângeri speciale pe distribuția pe ani.")


if __name__ == "__main__":
    main()
