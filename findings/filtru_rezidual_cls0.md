# Filtru rezidual cls0 — raport

**Scope:** elimină categorii minore de zgomot rămase după curățarea
cookies v3. Pașii de deduplicare și filtrare lungime vin separat.

## 1. Rezumat operații

| Operație | Număr propoziții |
|---|---|
| Input (din no_cookies v3) | 5,976 |
| Nemodificate | 5,908 |
| Aruncat — link Citește și | 3 |
| Aruncat — etichetă vorbitor | 50 |
| Aruncat — degenerat alfanumeric | 8 |
| Curățat — prefix Foto: | 7 |
| Aruncat — rest prea scurt după Foto: | 0 |
| **Output** | **5,915** |

**Retenție:** 98.98%

## 2. Breakdown pe sursă

| Sursă | Input | Output | Retenție |
|---|---|---|---|
| digi24.ro | 3,226 | 3,205 | 99.35% |
| g4media.ro | 2,750 | 2,710 | 98.55% |

## 3. Exemple — link-uri Citește și

- [digi24.ro] *9w*: Citește și De ce este importantă Transnistria pentru Rusia.
- [digi24.ro] *16w*: Citește și: Lecțiile Ucrainei pentru România și NATO, după șase luni de război total cu Rusia.
- [digi24.ro] *8w*: Citește și: Șeful Pentagonului respinge cererea lui Zelenski.

## 4. Exemple — etichete vorbitor

- [digi24.ro] *2w*: Imagini șocante:
- [g4media.ro] *4w*: „Aceasta este responsabilitatea mea”:
- [g4media.ro] *4w*: Dezinformări în conflictul Israel-Hamas:
- [g4media.ro] *1w*: Zelenski:
- [digi24.ro] *4w*: Institutul pentru Studiul Războiului:

## 5. Exemple — degenerate alfanumerice

- [digi24.ro] *1w*: '.'
- [g4media.ro] *1w*: 'ro.'
- [g4media.ro] *2w*: '– „NG”).'
- [digi24.ro] *1w*: 'Da.'
- [digi24.ro] *3w*: 'Asta e, f..!”'

## 6. Exemple — prefix Foto: curățat (ÎNAINTE / DUPĂ)

> Verifică vizual că bucata păstrată e conținut jurnalistic real.

**Sursă:** digi24.ro (24w → 19w)
- ÎNAINTE: Foto: highlandsystems.me Deschide galeria foto Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- DUPĂ:   Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.

**Sursă:** digi24.ro (21w → 19w)
- ÎNAINTE: Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- DUPĂ:   Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.

**Sursă:** digi24.ro (21w → 19w)
- ÎNAINTE: Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- DUPĂ:   Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.

**Sursă:** digi24.ro (21w → 19w)
- ÎNAINTE: Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- DUPĂ:   Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.

**Sursă:** digi24.ro (21w → 19w)
- ÎNAINTE: Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- DUPĂ:   Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.

## 7. Pași următori

1. **Deduplicare** pe hash normalizat — va elimina ~150 propoziții duplicate
   (inclusiv cele 5 identice 'Foto: highlandsystems.me Submarinul Kronos...').
2. **Filtrare lungime** pe percentilele [p5, p95] recalculate pe corpusul deduplicat.
3. **Embeddings** pe corpusul final.