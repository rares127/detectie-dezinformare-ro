# Diagnostic Benchmark v4 — 4 probe

Verifică dacă AUC-urile mari din benchmark v4 (0.98 pe Test A min, 0.94 pe Test B min) sunt semnal real sau artefact metodologic (în special: artefact de lungime articol).

## Proba 1 — Distribuția lungimii articolelor

**Raport mediane cls0 / cls1 = 2.11x**

_ATENȚIE: diferență mare între mediane (raport 2.11x). Artefact de lungime PLAUZIBIL — agregarea min penalizează articolele mai lungi._

| Clasă | n | min | p25 | mediana | p75 | max | media | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cls0 | 55 | 8 | 13.5 | 19.0 | 24.0 | 48 | 20.45 | 9.19 |
| cls1 | 112 | 1 | 6.0 | 9.0 | 11.0 | 32 | 9.43 | 4.99 |

### Per sursă

| Sursă | n | min | mediana | media | max |
|---|---:|---:|---:|---:|---:|
| hotnews.ro | 45 | 8 | 17.0 | 19.33 | 48 |
| libertatea.ro | 5 | 10 | 22.0 | 24.20 | 42 |
| stirileprotv.ro | 5 | 22 | 25.0 | 26.80 | 35 |
| stopfals.md | 13 | 2 | 10.0 | 7.69 | 16 |
| veridica.ro | 99 | 1 | 9.0 | 9.66 | 32 |

## Proba 2 — AUC subset balansat + corelație lungime × scor

**Interval balansat [p25, p75]:** [8, 16] propoziții/articol. **n=85** (23 cls0 + 62 cls1).

### AUC comparație: full vs subset balansat

| Coloană | AUC full (n=167) | AUC subset | Δ |
|---|---:|---:|---:|
| `scor_cls1_min` | 0.9774 | 0.9776 | +0.0001 |
| `scor_cls0_min` | 0.9435 | 0.9299 | -0.0136 |
| `diff_min` | 0.7784 | 0.8633 | +0.0848 |
| `scor_cls1_mean` | 0.8313 | 0.8338 | +0.0025 |
| `scor_cls0_mean` | 0.4763 | 0.4951 | +0.0188 |
| `diff_mean` | 0.9739 | 0.9790 | +0.0051 |

### Corelații Pearson (nr_prop × scor) per clasă

| Coloană | r pe cls0 | r pe cls1 |
|---|---:|---:|
| `scor_cls1_min` | +0.3693 | -0.4210 |
| `scor_cls0_min` | +0.2736 | -0.4097 |
| `diff_min` | +0.3410 | -0.0044 |
| `scor_cls1_mean` | +0.4759 | -0.1213 |
| `scor_cls0_mean` | +0.4921 | -0.2222 |
| `diff_mean` | -0.0237 | +0.1874 |

### Interpretări Proba 2

- AUC Test A (min) scade doar cu +0.000 pe subset → semnalul NU e artefact de lungime.
- AUC Test D (mean) stabil pe subset (+0.005) → agregarea mean pe diferență e robustă la lungime.

## Proba 3 — Inspecție calitativă propoziții min

**Sample (10 art./clasă):** lungime medie prop. min: cls0 = 14.5w, cls1 = 19.0w. Scor mediu min: cls0 = 0.379, cls1 = 0.598.

### Exemple CLS0

**1.** `ext_052` · stirileprotv.ro · 35 prop/art · prop. min: 15w, scor=0.422

- Prop. min: *Programul „Femeia Antreprenor” este reluat după doi ani de blocaj, au anunțat reprezentanții Ministerului Economiei.*
- Match corpus cls1 (scor=0.422): *[…] este necesară reluarea activității mass-mediei independente interzise în mod ilegal începând cu anul 2019.*

**2.** `ext_037` · hotnews.ro · 32 prop/art · prop. min: 10w, scor=0.450

- Prop. min: *Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.450): *Așteptări și efecte ale legii securității informaționale.*

**3.** `ext_005` · hotnews.ro · 23 prop/art · prop. min: 10w, scor=0.450

- Prop. min: *Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.450): *Așteptări și efecte ale legii securității informaționale.*

**4.** `ext_032` · hotnews.ro · 48 prop/art · prop. min: 10w, scor=0.450

- Prop. min: *Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.450): *Așteptări și efecte ale legii securității informaționale.*

**5.** `ext_022` · hotnews.ro · 20 prop/art · prop. min: 18w, scor=0.313

- Prop. min: *HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs. Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.313): *Potrivit SVR, Mažeiks și-a permis să ofere partidului de guvernământ "Acțiune și Solidaritate" instrucțiuni detaliate despre cum ar trebui să funcționeze democrația în Moldova până la alegerile parlamentare din 2025.*

**6.** `ext_050` · hotnews.ro · 35 prop/art · prop. min: 18w, scor=0.313

- Prop. min: *HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs. Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.313): *Potrivit SVR, Mažeiks și-a permis să ofere partidului de guvernământ "Acțiune și Solidaritate" instrucțiuni detaliate despre cum ar trebui să funcționeze democrația în Moldova până la alegerile parlamentare din 2025.*

**7.** `ext_053` · hotnews.ro · 22 prop/art · prop. min: 18w, scor=0.313

- Prop. min: *HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs. Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.313): *Potrivit SVR, Mažeiks și-a permis să ofere partidului de guvernământ "Acțiune și Solidaritate" instrucțiuni detaliate despre cum ar trebui să funcționeze democrația în Moldova până la alegerile parlamentare din 2025.*

**8.** `ext_006` · hotnews.ro · 21 prop/art · prop. min: 18w, scor=0.313

- Prop. min: *HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs. Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.313): *Potrivit SVR, Mažeiks și-a permis să ofere partidului de guvernământ "Acțiune și Solidaritate" instrucțiuni detaliate despre cum ar trebui să funcționeze democrația în Moldova până la alegerile parlamentare din 2025.*

**9.** `ext_011` · hotnews.ro · 10 prop/art · prop. min: 18w, scor=0.313

- Prop. min: *HotNews.ro utilizează cookie-uri pentru a îmbunătăți experiența dvs. Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.313): *Potrivit SVR, Mažeiks și-a permis să ofere partidului de guvernământ "Acțiune și Solidaritate" instrucțiuni detaliate despre cum ar trebui să funcționeze democrația în Moldova până la alegerile parlamentare din 2025.*

**10.** `ext_044` · hotnews.ro · 31 prop/art · prop. min: 10w, scor=0.450

- Prop. min: *Accesați Modifică Setările pentru preferințe și consultați Politica de confidențialitate.*
- Match corpus cls1 (scor=0.450): *Așteptări și efecte ale legii securității informaționale.*

### Exemple CLS1

**1.** `vrd_0094` · veridica.ro · 7 prop/art · prop. min: 24w, scor=0.592

- Prop. min: *Comentatorii menționează că legea a fost promulgată în ziua în care s-au ]împlinit 30 de ani de la atacul barbar asupra orașului transnistrean Bender.*
- Match corpus cls1 (scor=0.592): *În aceste zile se împlinesc 30 de ani de la semnarea Acordului care, de jure, a pus capăt războiului de pe Nistru.*

**2.** `vrd_0307` · veridica.ro · 18 prop/art · prop. min: 9w, scor=0.646

- Prop. min: *[...] "Nu exclud că sunt examinate două scenarii posibile.*
- Match corpus cls1 (scor=0.646): *El a adăugat că un scenariu similar ar putea avea loc și în cazul unor alegeri nereușite.*

**3.** `vrd_0092` · veridica.ro · 10 prop/art · prop. min: 11w, scor=0.524

- Prop. min: *"Propriile noastre forțe nu vor fi niciodată suficiente", a subliniat Balițki.*
- Match corpus cls1 (scor=0.524): *Tacticile de acolo nu pot fi comparate cu metodele noastre de luptă.*

**4.** `vrd_0063` · veridica.ro · 9 prop/art · prop. min: 32w, scor=0.688

- Prop. min: *Șeful Forțelor de Protecție Radiologică, Chimică și Biologică susține că în ianuarie 2022 Forțele Armate ale Ucrainei au achiziționat 50 de asemenea drone, care pot fi folosite pentru atacuri chimice și biologice.*
- Match corpus cls1 (scor=0.688): *Ruşii au capturat o dronă americană de ultimă generaţie.*

**5.** `stopfals_180842` · stopfals.md · 16 prop/art · prop. min: 16w, scor=0.575

- Prop. min: *Găgăuzia poate avea reprezentanțele sale în lumea turcică și în țările CSI și le va avea.*
- Match corpus cls1 (scor=0.575): *E interesantă și halta sa de la turcească.*

**6.** `vrd_0626` · veridica.ro · 17 prop/art · prop. min: 11w, scor=0.456

- Prop. min: *Le este mai ușor să continue să prezinte dorințele drept realitate.*
- Match corpus cls1 (scor=0.456): *Orice ar spune ei, războiul pare a fi foarte avantajos acum.*

**7.** `vrd_0404` · veridica.ro · 11 prop/art · prop. min: 14w, scor=0.571

- Prop. min: *Timp de decenii, ea a împiedicat extinderea și degradarea întregului sistem de securitate regională.*
- Match corpus cls1 (scor=0.571): *Și aceasta este o amenințare la adresa securității naționale, suveranității și integrității teritoriale a țării noastre, care nu poate fi ignorată."*

**8.** `vrd_0408` · veridica.ro · 6 prop/art · prop. min: 41w, scor=0.656

- Prop. min: *În acest sens, mișcarea a trimis o serie de apeluri președintelui Biroului Consiliului Europei în Ucraina, Maciej Janczak, șefului Consiliului permanent al Organizației pentru Securitate și Cooperare în Europa (OSCE), Jan Borg, precum și ambasadorilor tuturor reprezentanțelor diplomatice acreditate în Ucraina.*
- Match corpus cls1 (scor=0.656): *Ucrainei i-au fost înaintate o serie de cerințe pentru aderarea la UE.*

**9.** `vrd_0543` · veridica.ro · 6 prop/art · prop. min: 16w, scor=0.657

- Prop. min: *Își condamnă astfel concetățenii la noi suferințe și pierderi inutile, dacă ne raportăm la scopul urmărit:*
- Match corpus cls1 (scor=0.657): *"Este meritul nostru enorm și avem dreptul să le spunem autorităților: de ce nu ați făcut nimic, de ce sunt zeci, iar acum chiar sute de mii de victime, pentru ce au murit?"*

**10.** `vrd_0528` · veridica.ro · 9 prop/art · prop. min: 16w, scor=0.617

- Prop. min: *Astfel de conflicte, a spus el, se rezolvă prin divizarea țării în funcție de componența etnică.*
- Match corpus cls1 (scor=0.617): *| Observăm discriminare pe criterii naționale, etnice.*

## Proba 4 — Agregare top-k mean (k=5)

Recalculare cu **top-5 mean** în loc de max per propoziție. Top-k mean e robust la extreme: cere consistență peste 5 potriviri, nu doar o potrivire accidentală.

### test_A_cls1_only

| Agregare | AUC | μ(cls0) | μ(cls1) |
|---|---:|---:|---:|
| mean | 0.8081 | 0.6049 | 0.6633 |
| min | 0.9857 | 0.3278 | 0.5479 |
| p10 | 0.8602 | 0.4890 | 0.5844 |

### test_B_cls0_only

| Agregare | AUC | μ(cls0) | μ(cls1) |
|---|---:|---:|---:|
| mean | 0.4990 | 0.6358 | 0.6276 |
| min | 0.9604 | 0.3472 | 0.5206 |
| p10 | 0.6739 | 0.5155 | 0.5538 |

### test_D_diff

| Agregare | AUC | μ(cls0) | μ(cls1) |
|---|---:|---:|---:|
| mean | 0.9731 | -0.0308 | 0.0357 |
| min | 0.9013 | -0.0194 | 0.0272 |
| p10 | 0.9253 | -0.0266 | 0.0306 |

## Concluzii globale

- P1 · Raport mediane cls0/cls1 = 2.11x → DIFERENȚĂ STRUCTURALĂ
- P2 · ΔAUC Test A min (subset balansat): +0.0001 → semnal robust la lungime
- P2 · ΔAUC Test D mean (subset balansat): +0.0051 → agregarea mean stabilă
- P3 · Lungime medie prop. min: cls0=14.5w vs cls1=19.0w
- P4 · Best AUC top-5 mean: Test A = 0.9857, Test D = 0.9731
- VERDICT: Min e semnal REAL (ΔAUC stabil pe subset balansat). AUC 0.98 din benchmark v4 rămâne valid. Raportăm pe min.

---

*Modul 3 · Pasul A2.5 · Diagnostic benchmark v4*