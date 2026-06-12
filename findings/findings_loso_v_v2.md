# Findings — LOSO experiment (held-out: veridica.ro)

## Context metodologic

Scop: discrimina între:
- **(a)** model cu shortcut de sursă stilistic
- **(b)** model care învață genuin dezinformarea

Train: toate sursele EXCEPT `veridica.ro` → Test: TOT `veridica.ro`

## Setup

- **n_train (intern)**: 698
- **n_val (intern, early stopping)**: 124
- **n_test (held-out)**: 661

**Distribuție surse TRAIN:**

| Sursa | n |
|-------|---|
| digi24.ro | 330 |
| g4media.ro | 296 |
| stopfals.md | 72 |

## Rezultate

- **Accuracy**: 0.2935
- **Recall cls1**: 0.2935
- **Mean prob_cls1**: 0.2082
- **Median prob_cls1**: 0.0065
- **Corecte**: 194/661
- **Greșite**: 467/661

## Comparație cu baseline v2

- Baseline v2 (test full) recall cls1: **1.0000**
- LOSO recall cls1: **0.2935**
- **Drop**: +0.7065 (+70.65pp)

### Interpretare

Drop >30pp — SHORTCUT DE SURSĂ CONFIRMAT. Modelul învață predominant stilul veridica.ro, nu dezinformarea.

### Praguri (din handoff, Secțiunea 6.4):
- Drop <10pp → model învață genuin
- Drop 10-30pp → model învață mix dezinformare + stil sursei
- Drop >30pp → model învață predominant stil sursei

## Timp antrenare

135.8s (2.3 min)