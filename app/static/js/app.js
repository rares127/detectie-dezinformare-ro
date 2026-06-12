/**
 * Logica frontend pentru detectorul de dezinformare.
 *
 * Strategie de afisare diferentiata pe clasa (decizia arhitecturala principala):
 * - Pentru cls0 (credibil)  → afisam propozitii similare AMBELE corpus-uri
 *                             + buton "Genereaza explicatie LIME"
 * - Pentru cls1 (propaganda) → afisam propozitii similare AMBELE corpus-uri
 *                             SE ASCUNDE butonul LIME (justificat in nota metodologica)
 * - Pentru INCERT           → afisam doar verdict + baseline; fara propozitii/LIME
 *
 * Scorurile tehnice (prob_cls1, diff_mean, threshold etc.) NU mai sunt afisate
 * direct sub verdict — sunt grupate in panoul „Detaliile analizei" (buton toggle),
 * fiecare parametru cu o explicatie pe intelesul tuturor.
 *
 * Pragurile de colorare sunt derivate din threshold-ul de productie primit de la
 * backend (campul `threshold_productie` din /api/predict) — o singura sursa de
 * adevar; la o recalibrare, UI-ul se actualizeaza automat.
 *
 * Comunicare cu backend-ul:
 * - POST /api/predict      — rapid (~1-5s), timeout 60s
 * - POST /api/explain_lime — LENT (~10-30s), timeout 120s, doar pe cls0.
 *   Backend-ul recalculeaza decizia modulului 3 server-side (nu mai trimitem
 *   diff_mean de la client).
 */

// Stare globala minima (textul curent + ultima decizie — pentru LIME)
let stareCurenta = {
    text: null,
    decizie: null,     // 'dezinformare_pro_rusa' / 'stire_credibila' / 'incert'
    diff_mean: null,   // scorul modulului 3 din ultimul /predict (pentru boxul arhitectura)
    prob_cls1: null,   // P(dezinformare) al modulului 2 — folosit pentru boxul arhitectura duala
    threshold: null,   // threshold-ul de productie primit de la backend
};

// Fallback folosit doar inainte de primul raspuns al backend-ului
const THRESHOLD_FALLBACK = -0.0073;

// Timeouts fetch (ms) — peste estimarile normale, sub „inghetat la infinit"
const TIMEOUT_PREDICT_MS = 60_000;
const TIMEOUT_LIME_MS = 120_000;


/**
 * Praguri UI derivate din threshold-ul de productie.
 *
 * - band   = |τ| → zona neutra per propozitie: [−band, +band]
 * - intens = 4.1 × band (≈ ±0.03 la τ = −0.0073) — prag VIZUAL pentru
 *   colorarea intensa; nu e decizional, doar gradeaza intensitatea culorii.
 */
function praguriUI() {
    const tau = stareCurenta.threshold ?? THRESHOLD_FALLBACK;
    const band = Math.abs(tau);
    const intens = +(4.1 * band).toFixed(3);
    return { band, intens };
}


// ─────────────────────────────────────────────────────────────────────────────
// Counter caractere (live)
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('textInput').addEventListener('input', (e) => {
    const n = e.target.value.length;
    document.getElementById('charCount').textContent =
        `${n.toLocaleString('ro-RO')} caractere`;
});


// ─────────────────────────────────────────────────────────────────────────────
// Helper: arata/ascunde elemente
// ─────────────────────────────────────────────────────────────────────────────
function arata(id) { document.getElementById(id).classList.remove('hidden'); }
function ascunde(id) { document.getElementById(id).classList.add('hidden'); }

/** Mesaj prietenos pentru erorile de timeout/abort ale fetch-ului. */
function mesajEroareFetch(e, ctx) {
    if (e && (e.name === 'TimeoutError' || e.name === 'AbortError')) {
        return `${ctx} durează neobișnuit de mult și a fost întreruptă. ` +
               `Verificați că serverul rulează și încercați din nou.`;
    }
    return e.message || 'Eroare necunoscută.';
}


// ─────────────────────────────────────────────────────────────────────────────
// Actiunea principala: analizeaza articolul (POST /api/predict)
// ─────────────────────────────────────────────────────────────────────────────
async function analizeazaArticol() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) {
        afiseazaEroare('Introduceți un text înainte de a analiza.');
        return;
    }

    // Reset UI
    ascunde('rezultat');
    ascunde('errorBox');
    ascunde('rezultatLime');
    ascunde('lineLimeBox');
    ascunde('detaliiTehnice');
    arata('loadingPredict');
    document.getElementById('btnAnalizeaza').disabled = true;

    try {
        const resp = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
            signal: AbortSignal.timeout(TIMEOUT_PREDICT_MS),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Eroare HTTP ${resp.status}`);
        }

        const data = await resp.json();
        stareCurenta.text     = text;
        stareCurenta.decizie  = data.decizie;
        stareCurenta.diff_mean = data.scor_modul3_diff_mean;
        stareCurenta.prob_cls1 = data.scor_baseline_prob_cls1;
        // Threshold-ul de productie — sursa de adevar pentru pragurile UI
        stareCurenta.threshold = data['threshold_producție'] ?? stareCurenta.threshold;
        afiseazaRezultat(data);
    } catch (e) {
        afiseazaEroare(mesajEroareFetch(e, 'Analiza articolului'));
    } finally {
        ascunde('loadingPredict');
        document.getElementById('btnAnalizeaza').disabled = false;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Render rezultat principal — diferentiat pe clasa
// ─────────────────────────────────────────────────────────────────────────────
function afiseazaRezultat(data) {
    // 1. Verdict principal — culoare + text
    const verdictBox = document.getElementById('verdictBox');
    const verdictText = document.getElementById('verdictText');
    verdictText.textContent = data.decizie_display;

    // Curatam clase precedente
    verdictBox.className = 'rounded-lg p-6 border-2 text-center';
    verdictText.className = 'text-3xl font-bold mb-3';
    if (data.is_borderline) {
        verdictBox.classList.add('bg-amber-50', 'border-amber-400');
        verdictText.classList.add('text-amber-900');
    } else if (data.decizie === 'dezinformare_pro_rusa') {
        verdictBox.classList.add('bg-red-50', 'border-red-400');
        verdictText.classList.add('text-red-900');
    } else if (data.decizie === 'stire_credibila') {
        verdictBox.classList.add('bg-emerald-50', 'border-emerald-400');
        verdictText.classList.add('text-emerald-900');
    } else {  // incert
        verdictBox.classList.add('bg-slate-50', 'border-slate-400');
        verdictText.classList.add('text-slate-700');
    }

    // 2. Bara de incredere (doar daca nu e incert)
    if (data.decizie_incerta) {
        document.getElementById('incredereBox').style.display = 'none';
    } else {
        document.getElementById('incredereBox').style.display = 'block';
        const pct = (data.incredere * 100).toFixed(0);
        document.getElementById('increderePct').textContent = `${pct}% propagandă`;

        // Tooltip-ul markerului de threshold — valoarea reala de la backend
        document.getElementById('thresholdMarker').title =
            `Threshold = ${formatDiff(stareCurenta.threshold ?? THRESHOLD_FALLBACK)}`;

        // Bara de fill — culoare in functie de decizie
        const bara = document.getElementById('incredereFill');
        bara.style.width = `${pct}%`;
        bara.className = 'h-full transition-all duration-500 ' + (
            data.is_borderline             ? 'bg-amber-400' :
            data.decizie === 'dezinformare_pro_rusa' ? 'bg-red-500' : 'bg-emerald-500'
        );
    }

    // 3. Nota metodologica
    document.getElementById('notaMetodologica').textContent = data.nota_metodologica;

    // 4. Warning trunchiere
    if (data.metadata.input_truncat_xlmr) {
        arata('warningTrunchiere');
    } else {
        ascunde('warningTrunchiere');
    }

    // 5. Panou „Detaliile analizei" — parametrii tehnici, fiecare explicat.
    //    Panoul porneste inchis la fiecare analiza noua.
    renderDetaliiTehnice(data);
    resetToggleDetalii();

    // 6. STRATEGIA DIFERENTIATA PE CLASA
    //    Buton LIME doar pe cls0 (justificat empiric in nota_metodologica)
    if (data.decizie === 'stire_credibila') {
        arata('lineLimeBox');
    } else {
        ascunde('lineLimeBox');
    }

    // 7. Vizualizare colorata per propozitie — afisata MEREU daca nu e incert
    if (!data.decizie_incerta && data.propozitii_detalii && data.propozitii_detalii.length > 0) {
        actualizeazaLegenda();
        renderTextColorat(data.propozitii_detalii);
    } else {
        ascunde('textColoratBox');
    }

    arata('rezultat');
    document.getElementById('rezultat').scrollIntoView({ behavior: 'smooth' });
}


// ─────────────────────────────────────────────────────────────────────────────
// Panoul „Detaliile analizei" — parametrii tehnici explicati
// ─────────────────────────────────────────────────────────────────────────────

/** Formateaza un diff/threshold cu semn explicit si 4 zecimale. */
function formatDiff(v) {
    return (v >= 0 ? '+' : '') + v.toFixed(4);
}

/** Un rand din panoul de detalii: eticheta + valoare + explicatie simpla. */
function randDetaliu(eticheta, valoare, explicatie) {
    return `
        <div class="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div class="flex items-baseline justify-between gap-3 flex-wrap">
                <span class="text-sm font-semibold text-slate-800">${eticheta}</span>
                <span class="font-mono text-sm font-bold text-slate-900">${valoare}</span>
            </div>
            <p class="text-xs text-slate-500 mt-1 leading-relaxed">${explicatie}</p>
        </div>`;
}

/** Text prietenos pentru fiecare motiv de verdict borderline. */
function descriereMotivBorderline(motiv) {
    switch (motiv) {
        case 'dezacord_m2_cls1_m3_cls0':
            return 'Dezacord între module: analiza semantică (Modulul 3) indică „credibil", ' +
                   'dar clasificatorul global (Modulul 2) indică dezinformare cu probabilitate ' +
                   'mare. Tipic pentru articole care citează declarații propagandistice — ' +
                   'verificați manual.';
        case 'dezacord_m3_cls1_m2_cls0':
            return 'Dezacord între module: analiza semantică (Modulul 3) indică „dezinformare", ' +
                   'dar clasificatorul global (Modulul 2) indică probabilitate mică. Articolul ' +
                   'folosește probabil vocabular tematic (război, Rusia, NATO) într-un cadru legitim.';
        case 'proximitate_threshold':
            return 'Scorul semantic e foarte aproape de pragul de decizie — diferența e prea ' +
                   'mică pentru un verdict ferm.';
        case 'esantion_mic':
            return 'Prea puține propoziții valide pentru un verdict stabil — scorul semantic ' +
                   'e calculat pe un eșantion foarte mic.';
        default:
            return 'Verdict cu încredere redusă.';
    }
}

/**
 * Construieste randurile panoului de detalii pentru raspunsul curent.
 * Fiecare parametru tehnic primeste o explicatie pe intelesul tuturor.
 */
function renderDetaliiTehnice(data) {
    const m = data.metadata;
    const randuri = [];

    // Motivul borderline — primul, daca exista (cel mai important de inteles)
    if (data.is_borderline && data.motiv_borderline) {
        randuri.push(randDetaliu(
            '⚠️ De ce e verdictul incert?',
            '',
            escapeHtml(descriereMotivBorderline(data.motiv_borderline))
        ));
    }

    // Modulul 2 — probabilitate dezinformare
    randuri.push(randDetaliu(
        'Probabilitate dezinformare — Modulul 2 (XLM-RoBERTa)',
        (data.scor_baseline_prob_cls1 * 100).toFixed(1) + '%',
        'Cât de probabil consideră clasificatorul global că articolul e dezinformare. ' +
        'E un semnal orientativ: modelul poate fi indus în eroare de vocabularul tematic ' +
        '(Rusia, război, NATO) chiar și în articole legitime — de aceea NU decide singur verdictul.'
    ));

    // Modulul 3 — scorul semantic (decizional)
    if (data.scor_modul3_diff_mean !== null) {
        randuri.push(randDetaliu(
            'Scor semantic — Modulul 3 (diff_mean)',
            formatDiff(data.scor_modul3_diff_mean),
            'Diferența medie dintre cât de mult seamănă propozițiile articolului cu corpusul ' +
            'de propagandă față de corpusul de presă credibilă. Valori negative = mai aproape ' +
            'de presa credibilă; valori pozitive = mai aproape de narațiunile propagandistice. ' +
            '<strong>Acesta este scorul care decide verdictul final.</strong>'
        ));

        // Similaritatile medii cu cele doua corpusuri
        randuri.push(randDetaliu(
            'Similaritate cu presa credibilă / cu propaganda',
            (data.scor_modul3_cls0_mean * 100).toFixed(1) + '% / ' +
            (data.scor_modul3_cls1_mean * 100).toFixed(1) + '%',
            'Pentru fiecare propoziție din articol se caută cea mai asemănătoare propoziție ' +
            'din cele două corpusuri de referință (câte 5.290 de propoziții fiecare); aici e ' +
            'media acestor potriviri. Diferența dintre cele două procente dă scorul semantic.'
        ));

        // Pragul de decizie
        const tau = stareCurenta.threshold ?? THRESHOLD_FALLBACK;
        randuri.push(randDetaliu(
            'Prag de decizie (threshold)',
            formatDiff(tau),
            'Pragul calibrat statistic (validare încrucișată 5-fold pe setul de test). ' +
            'Dacă scorul semantic depășește pragul → „Dezinformare pro-rusă"; ' +
            'dacă e sub prag → „Știre credibilă".'
        ));
    } else {
        randuri.push(randDetaliu(
            'Scor semantic — Modulul 3 (diff_mean)',
            'N/A',
            'Nu a putut fi calculat: articolul nu conține nicio propoziție cu 7-54 de cuvinte. ' +
            'De aceea verdictul este „Incert".'
        ));
    }

    // Propozitii analizate
    randuri.push(randDetaliu(
        'Propoziții analizate',
        `${m.n_propozitii_valide} din ${m.n_propozitii_total}`,
        'Analiza semantică folosește doar propozițiile cu 7-54 de cuvinte (identic cu ' +
        'calibrarea sistemului). Propozițiile prea scurte sau prea lungi sunt ignorate. ' +
        'Sub 3 propoziții valide, verdictul e marcat ca având încredere redusă.'
    ));

    // Trunchiere XLM-R
    randuri.push(randDetaliu(
        'Text trunchiat pentru Modulul 2',
        m.input_truncat_xlmr ? 'Da' : 'Nu',
        m.input_truncat_xlmr
            ? 'Textul depășește 256 tokens (~1500 caractere), deci clasificatorul global a ' +
              'analizat doar începutul. Analiza semantică (Modulul 3) nu e afectată — ' +
              'lucrează pe fiecare propoziție în parte.'
            : 'Textul încape integral în fereastra de 256 tokens a clasificatorului global.'
    ));

    // Timp de analiza
    const sec = (m.timp_inferenta_ms / 1000).toFixed(1);
    randuri.push(randDetaliu(
        'Timp de analiză',
        `${m.timp_inferenta_ms.toLocaleString('ro-RO')} ms (~${sec}s)`,
        'Include segmentarea în propoziții, clasificarea globală și compararea fiecărei ' +
        'propoziții cu cele 10.580 de propoziții de referință.'
    ));

    document.getElementById('detaliiContent').innerHTML = randuri.join('');
}

/** Toggle pentru panoul de detalii tehnice. */
function toggleDetalii() {
    const panou = document.getElementById('detaliiTehnice');
    const chevron = document.getElementById('chevronDetalii');
    if (panou.classList.contains('hidden')) {
        arata('detaliiTehnice');
        chevron.style.transform = 'rotate(180deg)';
    } else {
        ascunde('detaliiTehnice');
        chevron.style.transform = '';
    }
}

/** Reseteaza toggle-ul la starea inchisa (la fiecare analiza noua). */
function resetToggleDetalii() {
    ascunde('detaliiTehnice');
    document.getElementById('chevronDetalii').style.transform = '';
}


// ─────────────────────────────────────────────────────────────────────────────
// Vizualizare colorata per propozitie (Modul 3)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Mapeaza diff-ul per propozitie la o culoare CSS de fundal.
 * Pragurile sunt derivate din threshold-ul de productie (vezi praguriUI).
 */
function culoarePentruDiff(diff) {
    const { band, intens } = praguriUI();
    if (diff > intens)   return 'rgba(220, 38, 38, 0.22)';      // rosu intens — propaganda certa
    if (diff > band)     return 'rgba(252, 165, 165, 0.45)';    // rosu deschis — suspect
    if (diff >= -band)   return 'rgba(203, 213, 225, 0.45)';    // gri — borderline
    if (diff >= -intens) return 'rgba(134, 239, 172, 0.50)';    // verde deschis — probabil credibil
    return 'rgba(22, 163, 74, 0.28)';                            // verde intens — clar credibil
}

/**
 * Genereaza mesajul explicativ afisat in panoul de detalii.
 * Zona neutra: [−band, +band], derivata din threshold.
 */
function mesajExplicativPropozitie(diff) {
    const { band } = praguriUI();
    if (diff > band) {
        return 'Această propoziție seamănă mai mult cu narațiuni propagandistice din corpus.';
    }
    if (diff < -band) {
        return 'Această propoziție seamănă mai mult cu presa credibilă din corpus.';
    }
    return 'Propoziție neutră — similaritate apropiată cu ambele corpusuri.';
}

/**
 * Completeaza intervalele numerice din legenda pe baza pragurilor derivate
 * din threshold-ul de productie — o singura sursa de adevar cu backend-ul.
 */
function actualizeazaLegenda() {
    const { band, intens } = praguriUI();
    const fb = band.toFixed(4);
    const fi = intens.toFixed(3);
    document.getElementById('legCerta').textContent    = `Propagandă certă (diff > +${fi})`;
    document.getElementById('legSuspect').textContent  = `Suspect (+${fb} … +${fi})`;
    document.getElementById('legNeutru').textContent   = `Neutru / borderline (−${fb} … +${fb})`;
    document.getElementById('legProbabil').textContent = `Probabil credibil (−${fi} … −${fb})`;
    document.getElementById('legClar').textContent     = `Clar credibil (diff < −${fi})`;
}

/**
 * Randeaza textul articolului ca bloc continuu cu propozitii colorate.
 * Click pe o propozitie → afiseaza panoul de detalii dedesubt.
 */
function renderTextColorat(propozitiiDetalii) {
    const container = document.getElementById('textColoratContent');
    container.innerHTML = '';
    ascunde('panouDetalii');

    // Indice propozitie selectata — gestionat in closure pentru reset corect
    let idxSelectat = null;

    propozitiiDetalii.forEach((prop, idx) => {
        const span = document.createElement('span');
        span.className = 'prop-colorata';
        span.style.backgroundColor = culoarePentruDiff(prop.diff);
        span.textContent = prop.text;
        // Tooltip rapid cu valoarea numerica
        span.title = `diff = ${formatDiff(prop.diff)}`;

        span.addEventListener('click', () => {
            if (idxSelectat === idx) {
                // A doua apasare pe aceeasi propozitie → inchide panoul
                span.classList.remove('prop-selectata');
                ascunde('panouDetalii');
                idxSelectat = null;
                return;
            }
            // Dezelecteaza propozitia anterioara
            const anterioară = container.querySelector('.prop-selectata');
            if (anterioară) anterioară.classList.remove('prop-selectata');

            span.classList.add('prop-selectata');
            idxSelectat = idx;
            afiseazaDetaliiPropozitie(prop);
            arata('panouDetalii');
        });

        container.appendChild(span);
        // Spatiu separator intre propozitii (text node simplu)
        container.appendChild(document.createTextNode(' '));
    });

    arata('textColoratBox');
}

/**
 * Populeaza panoul de detalii pentru propozitia selectata.
 */
function afiseazaDetaliiPropozitie(prop) {
    const panou = document.getElementById('panouDetalii');

    const diffStr = formatDiff(prop.diff);
    const cls0Pct = (prop.scor_cls0 * 100).toFixed(1);
    const cls1Pct = (prop.scor_cls1 * 100).toFixed(1);
    const sim0Pct = (prop.match_cls0_sim * 100).toFixed(0);
    const sim1Pct = (prop.match_cls1_sim * 100).toFixed(0);
    const mesaj   = mesajExplicativPropozitie(prop.diff);

    panou.innerHTML = `
        <div class="p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-3 text-sm">

            <!-- Scoruri numerice -->
            <div class="flex flex-wrap gap-4 font-mono text-xs text-slate-600
                        bg-white px-3 py-2 rounded border border-slate-200">
                <span>diff = <strong>${escapeHtml(diffStr)}</strong></span>
                <span>cls0 (credibil) = <strong>${cls0Pct}%</strong></span>
                <span>cls1 (propagandă) = <strong>${cls1Pct}%</strong></span>
            </div>

            <!-- Match corpusul credibil -->
            <div class="p-3 bg-emerald-50 border border-emerald-200 rounded">
                <p class="text-xs font-semibold text-emerald-800 mb-1">
                    🟢 Cel mai bun match din corpusul credibil —
                    ${sim0Pct}% similaritate ·
                    <span class="font-mono">${escapeHtml(prop.match_cls0_sursa)}</span>
                </p>
                <p class="text-slate-700 italic text-xs leading-relaxed">
                    „${escapeHtml(prop.match_cls0_text)}"
                </p>
            </div>

            <!-- Match corpusul propagandă -->
            <div class="p-3 bg-red-50 border border-red-200 rounded">
                <p class="text-xs font-semibold text-red-800 mb-1">
                    🔴 Cel mai bun match din corpusul propagandă —
                    ${sim1Pct}% similaritate ·
                    <span class="font-mono">${escapeHtml(prop.match_cls1_sursa)}</span>
                </p>
                <p class="text-slate-700 italic text-xs leading-relaxed">
                    „${escapeHtml(prop.match_cls1_text)}"
                </p>
            </div>

            <!-- Concluzie -->
            <p class="text-xs text-slate-500 italic">${escapeHtml(mesaj)}</p>
        </div>
    `;
}


// ─────────────────────────────────────────────────────────────────────────────
// Box explicativ: arhitectura duala M2 vs. M3 (apare conditionat in LIME)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Afiseaza (sau ascunde) boxul albastru care explica de ce apar cuvinte
 * „propagandistice" intr-o stire credibila.
 *
 * Conditie de afisare:
 *   - verdictul final al sistemului este cls0 (Stire credibila)
 *   - modulul 2 (XLM-R izolat) ar fi clasificat-o ca cls1 (prob_cls1 > 0.5)
 *
 * Acest caz ilustreaza exact valoarea arhitecturii duale fata de un singur
 * clasificator global — util in sectiunea de demonstrare din teza.
 *
 * @param {Array} cuvinteEvidentiate  Lista completa din raspunsul /explain_lime
 */
function afiseazaExplicatieArchitectura(cuvinteEvidentiate) {
    // Verificam conditia de afisare
    if (stareCurenta.decizie !== 'stire_credibila' || stareCurenta.prob_cls1 === null
            || stareCurenta.prob_cls1 <= 0.5) {
        ascunde('explicatieArchitectura');
        return;
    }

    // Top 3 cuvinte cu pondere negativa = semnal cel mai puternic spre cls1
    // (pondere negativa inseamna ca prezenta cuvantului impinge spre propaganda)
    const cuvinteRosii = cuvinteEvidentiate
        .filter(c => c.pondere < 0)
        .sort((a, b) => a.pondere - b.pondere)  // crescator → cel mai negativ primul
        .slice(0, 3);

    // Daca nu exista deloc cuvinte cu semnal cls1, nu afisam boxul
    if (cuvinteRosii.length === 0) {
        ascunde('explicatieArchitectura');
        return;
    }

    // Formatam lista de cuvinte intre ghilimele romanesti
    const listaCuvinte = cuvinteRosii
        .map(c => `„${escapeHtml(c.cuvant)}"`)
        .join(', ');

    const scorBaselinePct = (stareCurenta.prob_cls1 * 100).toFixed(1);

    // diff_mean formatat cu semn explicit
    const diffMeanStr = stareCurenta.diff_mean !== null
        ? formatDiff(stareCurenta.diff_mean)
        : 'N/A';

    const tauStr = formatDiff(stareCurenta.threshold ?? THRESHOLD_FALLBACK);

    // Construim textul explicativ cu valorile dinamice inserate
    document.getElementById('textExplicatieArh').innerHTML =
        `Modelul XLM-RoBERTa (Modulul 2) a identificat vocabular specific ` +
        `conflictului — termeni precum ${listaCuvinte} — care apar frecvent ` +
        `și în narațiunile propagandistice. Bazat exclusiv pe acest semnal ` +
        `lexical, clasificatorul global ar fi prezis dezinformare ` +
        `(<strong>P(cls1) = ${escapeHtml(scorBaselinePct)}%</strong>).` +
        `<br><br>` +
        `Modulul 3 (analiza semantică per propoziție) a corectat această decizie: ` +
        `propozițiile articolului seamănă structural mai mult cu presa credibilă ` +
        `din corpus (<strong>diff_mean = ${escapeHtml(diffMeanStr)}</strong>, ` +
        `față de pragul ${escapeHtml(tauStr)}). Sistemul combină ambele semnale — lexical și ` +
        `semantic — pentru a ajunge la verdictul final: <strong>Știre credibilă</strong>.`;

    arata('explicatieArchitectura');
}


// ─────────────────────────────────────────────────────────────────────────────
// Actiune secundara: genereaza LIME (POST /api/explain_lime)
// ─────────────────────────────────────────────────────────────────────────────
async function genereazaLime() {
    if (!stareCurenta.text) {
        afiseazaEroare('Niciun articol analizat. Apăsați mai întâi „Analizează articol".');
        return;
    }
    if (stareCurenta.decizie !== 'stire_credibila') {
        afiseazaEroare('LIME e disponibil doar pentru articole prezise ca credibile.');
        return;
    }

    arata('loadingLime');
    ascunde('rezultatLime');
    document.getElementById('btnLime').disabled = true;

    try {
        const resp = await fetch('/api/explain_lime', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Trimitem DOAR textul — backend-ul recalculeaza decizia modulului 3
            // server-side (eligibilitatea LIME nu mai depinde de date de la client)
            body: JSON.stringify({ text: stareCurenta.text }),
            signal: AbortSignal.timeout(TIMEOUT_LIME_MS),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Eroare HTTP ${resp.status}`);
        }

        const data = await resp.json();
        afiseazaRezultatLime(data);
    } catch (e) {
        afiseazaEroare(mesajEroareFetch(e, 'Generarea explicației LIME'));
    } finally {
        ascunde('loadingLime');
        document.getElementById('btnLime').disabled = false;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Render rezultat LIME — cuvinte colorate dupa pondere
// ─────────────────────────────────────────────────────────────────────────────
function afiseazaRezultatLime(data) {
    const container = document.getElementById('cuvinteEvidentiate');
    container.innerHTML = '';

    // Resetam boxul arhitectura duala — il reconstruim mai jos daca e cazul
    ascunde('explicatieArchitectura');

    // Pondere maxima absoluta — pentru normalizare opacitati
    const maxAbs = Math.max(...data.cuvinte_evidentiate.map(c => Math.abs(c.pondere)));

    data.cuvinte_evidentiate.forEach(c => {
        const intensitate = Math.abs(c.pondere) / maxAbs;  // ∈ [0, 1]
        const span = document.createElement('span');

        // Pondere POZITIVA = sprijina cls0 (credibil) → verde
        // Pondere NEGATIVA = sprijina cls1 (propaganda) → rosu
        const culoareBaza = c.pondere > 0 ? '16, 185, 129' : '220, 38, 38';
        const opacitate = 0.15 + 0.45 * intensitate;  // [0.15, 0.6]

        span.className = 'inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-sm font-medium';
        span.style.backgroundColor = `rgba(${culoareBaza}, ${opacitate})`;
        span.style.border = `1px solid rgba(${culoareBaza}, ${opacitate + 0.3})`;
        span.innerHTML = `
            <span>${escapeHtml(c.cuvant)}</span>
            <span class="text-xs opacity-70">${c.pondere >= 0 ? '+' : ''}${c.pondere.toFixed(3)}</span>
        `;
        container.appendChild(span);
    });

    document.getElementById('fidelityLime').textContent =
        `Fidelity LIME: ${data.fidelity_lime.toFixed(3)} (R² al modelului-surogat local)`;
    document.getElementById('notaValidareLime').textContent = data.nota_validare;

    // Box arhitectura duala — apare DOAR cand: cls0 final + M2 singur ar fi zis cls1
    afiseazaExplicatieArchitectura(data.cuvinte_evidentiate);

    arata('rezultatLime');
}


// ─────────────────────────────────────────────────────────────────────────────
// Eroare generica
// ─────────────────────────────────────────────────────────────────────────────
function afiseazaEroare(mesaj) {
    document.getElementById('errorMsg').textContent = mesaj;
    arata('errorBox');
}


// ─────────────────────────────────────────────────────────────────────────────
// Utilitar: escape HTML pentru a preveni XSS
// ─────────────────────────────────────────────────────────────────────────────
function escapeHtml(str) {
    if (typeof str !== 'string') return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
