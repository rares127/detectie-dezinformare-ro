# Findings — LOSO experiment (held-out: stopfals.md)

## Context metodologic

Scop: discrimina între:
- **(a)** model cu shortcut de sursă stilistic
- **(b)** model care învață genuin dezinformarea

Train: toate sursele EXCEPT `stopfals.md` → Test: TOT `stopfals.md`

## Setup

- **n_train (intern)**: 1188
- **n_val (intern, early stopping)**: 210
- **n_test (held-out)**: 85

**Distribuție surse TRAIN:**

| Sursa | n |
|-------|---|
| veridica.ro | 562 |
| digi24.ro | 330 |
| g4media.ro | 296 |

## Rezultate

- **Accuracy**: 0.9529
- **Recall cls1**: 0.9529
- **Mean prob_cls1**: 0.9518
- **Median prob_cls1**: 0.9997
- **Corecte**: 81/85
- **Greșite**: 4/85

## Comparație cu baseline v2

- Baseline v2 (test full) recall cls1: **1.0000**
- LOSO recall cls1: **0.9529**
- **Drop**: +0.0471 (+4.71pp)

### Interpretare

Drop <10pp — modelul învață GENUIN semnalul de dezinformare, nu depinde critic de sursa stopfals.md.

### Praguri (din handoff, Secțiunea 6.4):
- Drop <10pp → model învață genuin
- Drop 10-30pp → model învață mix dezinformare + stil sursei
- Drop >30pp → model învață predominant stil sursei

## Timp antrenare

231.9s (3.9 min)