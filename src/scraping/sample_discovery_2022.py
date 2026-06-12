"""
sample_discovery_2022.py — Sampling stratificat pe luna din discovery_digi24_v1_2022.jsonl

SCOP:
    Din JSONL-ul de discovery cu ~4662 articole 2022, selecteaza un sub-sample
    stratificat temporal (pe luna) pentru a evita fetch-ul a 2.5 ore pe articole
    din care vom folosi doar 213.

STRATEGIE:
    Problema: `data_listing` NU e populat in JSONL (vezi scraper_digi24_v1_2022.py,
    DiscoveryItem salveaza data_listing=""). Nu putem stratifica direct pe luna
    fara sa avem data articolului.

    Solutie: stratificam pe CAMPUL `pagina`. Digi24 listeaza descrescator
    cronologic, deci paginile ~80-100 = decembrie 2022, paginile ~160-180 =
    februarie 2022. Impartim intervalul de pagini ocupate de 2022 in N bucket-uri
    egale si samplem uniform din fiecare. Aproximativ echivalent cu stratificare
    lunara.

    Varianta imbunatatita (optionala): daca `data_listing` ar fi populata, am
    folosi-o direct. Aici mergem pe euristica paginarii.

PARAMETRI:
    --input       JSONL de input (default: data/raw/discovery_digi24_v1_2022.jsonl)
    --output      JSONL de output sample (default: discovery_digi24_v1_2022_sampled.jsonl)
    --n-target    Numar tinta de articole in sample (default: 400)
    --n-buckets   Numar de bucket-uri stratificate (default: 11, pentru 11 luni
                  active 2022: feb-dec, razboiul a inceput in 24.02.2022)
    --seed        Seed random (default: 42)

USAGE:
    python sample_discovery_2022.py --n-target 400
    # Apoi: redenumeste JSONL-urile si ruleaza fetch pe cel sample-at
    mv data/raw/discovery_digi24_v1_2022.jsonl data/raw/discovery_digi24_v1_2022_full.jsonl
    mv data/raw/discovery_digi24_v1_2022_sampled.jsonl data/raw/discovery_digi24_v1_2022.jsonl
    python scraper_digi24_v1_2022.py fetch
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sampling stratificat pe paginare din discovery JSONL 2022."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/discovery_digi24_v1_2022.jsonl"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/discovery_digi24_v1_2022_sampled.jsonl"),
    )
    ap.add_argument("--n-target", type=int, default=400)
    ap.add_argument("--n-buckets", type=int, default=11)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    if not args.input.exists():
        raise SystemExit(f"Input nu există: {args.input}")

    # Citesc toate intrarile
    items = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    print(f"Input: {len(items)} articole în {args.input.name}")

    # Distributie pe pagina
    pages = sorted({it["pagina"] for it in items})
    print(f"Pagini distincte: {len(pages)}, interval: {pages[0]}..{pages[-1]}")

    # Per tag_sursa
    per_tag = defaultdict(list)
    for it in items:
        per_tag[it["tag_sursa"]].append(it)
    print("Per tag:")
    for tag, lst in per_tag.items():
        tag_pages = sorted({it["pagina"] for it in lst})
        print(f"  {tag}: {len(lst)} articole, pagini {tag_pages[0]}..{tag_pages[-1]}")

    # ─── Stratificare pe bucket-uri de pagini ───
    # Impartim per tag separat, apoi consolidam
    # (fiecare tag are propriul lui spatiu de paginare)
    sampled = []
    target_per_tag = args.n_target // len(per_tag)

    for tag, lst in per_tag.items():
        tag_pages_sorted = sorted({it["pagina"] for it in lst})
        pmin, pmax = tag_pages_sorted[0], tag_pages_sorted[-1]
        nbuck = max(1, min(args.n_buckets, len(tag_pages_sorted)))
        # Fiecare bucket = un interval de pagini
        bucket_size = (pmax - pmin + 1) / nbuck
        buckets = defaultdict(list)
        for it in lst:
            b = int((it["pagina"] - pmin) / bucket_size)
            b = min(b, nbuck - 1)  # clamp pentru ultimul bucket
            buckets[b].append(it)

        # Sample uniform din fiecare bucket
        per_bucket = target_per_tag // nbuck
        extra = target_per_tag - per_bucket * nbuck  # ce ramane, distribuim primelor N
        print(
            f"\n  Tag {tag}: {nbuck} bucket-uri × ~{per_bucket} articole "
            f"(+{extra} extra pe primele bucket-uri)"
        )
        for b_idx in range(nbuck):
            n_wanted = per_bucket + (1 if b_idx < extra else 0)
            pool = buckets.get(b_idx, [])
            n_take = min(n_wanted, len(pool))
            if n_take < n_wanted:
                print(
                    f"    Bucket {b_idx}: pool doar {len(pool)}, vroiam {n_wanted} — iau tot"
                )
            chosen = random.sample(pool, n_take) if pool else []
            sampled.extend(chosen)

    print(f"\nTotal sampled: {len(sampled)} articole (țintă: {args.n_target})")

    # Dedup final pe id_articol (paranoia — tag-urile pot avea overlap)
    seen = set()
    deduped = []
    for it in sampled:
        if it["id_articol"] in seen:
            continue
        seen.add(it["id_articol"])
        deduped.append(it)
    if len(deduped) < len(sampled):
        print(f"După dedup pe id_articol: {len(deduped)} (eliminat {len(sampled)-len(deduped)})")

    # Scriu output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for it in deduped:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\n✅ Scris {len(deduped)} articole în {args.output}")

    # Sample preview: primele 3 titluri din fiecare tag
    print("\n── Preview: 3 titluri random per tag ──")
    per_tag_out = defaultdict(list)
    for it in deduped:
        per_tag_out[it["tag_sursa"]].append(it)
    for tag, lst in per_tag_out.items():
        print(f"\n  {tag}:")
        for it in random.sample(lst, min(3, len(lst))):
            print(f"    [p={it['pagina']}] {it['titlu_listing'][:80]}")


if __name__ == "__main__":
    main()
