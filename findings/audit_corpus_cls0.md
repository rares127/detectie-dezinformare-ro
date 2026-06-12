# Audit calitate corpus cls0

**Input:** 5,976 propoziții segmentate din articolele cls0 (Digi24 + G4Media)

## 1. Overview categorii de zgomot

| Categorie | Total | % din corpus | digi24 | g4media |
|---|---|---|---|---|
| `boilerplate_cms` | 10 | 0.17% | 10 | 0 |
| `eticheta_vorbitor` | 15 | 0.25% | 4 | 11 |
| `probabil_curat` | 5,944 | 99.46% | 3,206 | 2,738 |
| `titlu_scurt_probabil` | 2 | 0.03% | 2 | 0 |
| `titluri_concatenate` | 5 | 0.08% | 4 | 1 |

## 2. Boilerplate CMS — breakdown pe pattern

| Pattern | Matches | digi24 | g4media |
|---|---|---|---|
| `cookie_afisare` | 0 | 0 | 0 |
| `cookie_actualizare` | 0 | 0 | 0 |
| `cookie_accept` | 0 | 0 | 0 |
| `cookie_generic` | 0 | 0 | 0 |
| `continut_afisare` | 0 | 0 | 0 |
| `social_media_plugin` | 0 | 0 | 0 |
| `abonare_newsletter` | 0 | 0 | 0 |
| `citeste_si` | 3 | 3 | 0 |
| `foto_credit` | 7 | 7 | 0 |

### Exemple boilerplate detectat (primele 5 per pattern)

#### `citeste_si` — 3 matches total

- [digi24.ro] *9w*: Citește și De ce este importantă Transnistria pentru Rusia.
- [digi24.ro] *16w*: Citește și: Lecțiile Ucrainei pentru România și NATO, după șase luni de război total cu Rusia.
- [digi24.ro] *8w*: Citește și: Șeful Pentagonului respinge cererea lui Zelenski.

#### `foto_credit` — 7 matches total

- [digi24.ro] *24w*: Foto: highlandsystems.me Deschide galeria foto Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- [digi24.ro] *21w*: Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- [digi24.ro] *21w*: Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- [digi24.ro] *21w*: Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.
- [digi24.ro] *21w*: Foto: highlandsystems.me Submarinul Kronos, dezvoltat de o companie din Emiratele Arabe, pe care ucrainenii l-ar putea folosi în războiul cu Rusia.

## 3. Heuristici structurale

| Heuristică | Matches | % | digi24 | g4media |
|---|---|---|---|---|
| `fara_punct_final` | 55 | 0.92% | 11 | 44 |
| `incepe_cu_ghilimele` | 770 | 12.88% | 459 | 311 |
| `multe_ghilimele_interne` | 267 | 4.47% | 173 | 94 |
| `colon_fara_urmare` | 15 | 0.25% | 4 | 11 |
| `multi_caps` | 299 | 5.00% | 181 | 118 |
| `scurta_si_fara_punct` | 2 | 0.03% | 2 | 0 |

### Exemple — titluri concatenate (scor_suspect >= 2)

- [g4media.ro] *25w* [multe_ghilimele_interne, multi_caps]:
  > Într-un interviu acordat publicației The Guardian, generalul Eirik Kristoffersen a declarat că „armele nucleare” ale Rusiei sunt „singurul lucru” care „amenință cu adevărat Statele Unite”.

- [digi24.ro] *26w* [multe_ghilimele_interne, multi_caps]:
  > Președintele Franței Emmanuel Macron a spus despre negocierile „directe” propuse de președintele rus Vladimir Putin între Rusia şi Ucraina că reprezintă o „primă mișcare, dar insuficientă”.

- [digi24.ro] *15w* [multe_ghilimele_interne, multi_caps]:
  > Republica Moldova respinge acuzațiile Rusiei că Ucraina „vrea să invadeze Transnistria”: „Facem apel la calm”.

- [digi24.ro] *83w* [multe_ghilimele_interne, multi_caps]:
  > Secretarul general al NATO, Mark Rutte, a catalogat luni anunţul preşedintelui american Donald Trump, conform căruia Washingtonul va participa la garanţiile de securitate pentru Ucraina , în colaborare cu partenerii europeni, drept un „ pas mare ” şi

- [digi24.ro] *30w* [multe_ghilimele_interne, multi_caps]:
  > Ucraina cere Occidentului să adopte „sancțiuni severe” împotriva Rusiei În acelaşi timp, Ministerul de Externe de la Kiev a făcut apel marţi la Occident să adopte „sancţiuni severe” împotriva Rusiei.

## 4. Ipoteza „zgomot concentrat pe digi24"

- digi24.ro: **20 / 3,226** propoziții detectate ca zgomot (0.62%)
- g4media.ro: **12 / 2,750** propoziții detectate ca zgomot (0.44%)

> Ipoteza nu se confirmă net — zgomotul e distribuit comparabil pe ambele surse.

## 5. Eșantion stratificat pentru verificare manuală

> **Instrucțiuni pentru audit manual.** Citește fiecare propoziție și compară
> coloana `categorie_detectata` cu judecata ta. Ne interesează:
> - **False positives**: propoziții marcate ca zgomot dar care sunt conținut real
> - **False negatives**: propoziții marcate `probabil_curat` dar care sunt artefacte

### foarte_scurte (1-4 cuvinte)

- [g4media.ro] *1w* `probabil_curat` (scor=0):
  > ro.
- [digi24.ro] *4w* `probabil_curat` (scor=0):
  > „I-am împiedicat să acţioneze.
- [g4media.ro] *4w* `probabil_curat` (scor=0):
  > „Lumina va învinge întunericul.
- [g4media.ro] *3w* `probabil_curat` (scor=0):
  > România 🇷🇴 ❤️.
- [digi24.ro] *4w* `probabil_curat` (scor=0):
  > „Ne vânează, ne vânează!
- [g4media.ro] *3w* `probabil_curat` (scor=0):
  > Localnicii adună legumele.
- [digi24.ro] *4w* `probabil_curat` (scor=0):
  > Ursula von der Leyen:
- [digi24.ro] *3w* `titlu_scurt_probabil` (scor=1):
  > „Eram foarte agitați.”
- [digi24.ro] *3w* `probabil_curat` (scor=0):
  > „Bineînţeles, vom continua.
- [digi24.ro] *4w* `probabil_curat` (scor=0):
  > Institutul pentru Studiul Războiului:
- [digi24.ro] *4w* `probabil_curat` (scor=0):
  > "Plasa îi va opri.
- [digi24.ro] *4w* `probabil_curat` (scor=0):
  > Redut este noul Wagner.
- [digi24.ro] *2w* `probabil_curat` (scor=0):
  > O rușine.
- [digi24.ro] *3w* `probabil_curat` (scor=0):
  > Atunci ne afectează.
- [digi24.ro] *2w* `probabil_curat` (scor=0):
  > Reacția Rusiei.

### scurte (5-14 cuvinte)

- [digi24.ro] *11w* `probabil_curat` (scor=0):
  > O putem face altfel decât pe câmpul de luptă în Ucraina?
- [g4media.ro] *11w* `probabil_curat` (scor=0):
  > Kamala Harris: SUA au o prezență continuă și rotativă în România.
- [g4media.ro] *11w* `probabil_curat` (scor=0):
  > Mariam Mirakian, 25 de ani, a așteptat răbdătoare la semaforul roșu.
- [digi24.ro] *10w* `probabil_curat` (scor=0):
  > „Este ca și cum i-ai da mai multe gloanțe ucigașului”.
- [digi24.ro] *6w* `probabil_curat` (scor=0):
  > O singură armă le poate opri.
- [digi24.ro] *13w* `probabil_curat` (scor=0):
  > Cine este românul prins că spiona obiective militare NATO și transmitea informații Rusiei.
- [g4media.ro] *8w* `probabil_curat` (scor=0):
  > Există ţări care credeţi că ar fi indispensabile?
- [digi24.ro] *11w* `probabil_curat` (scor=0):
  > Andrei Kotenko are 20 de ani și voia să devină jurnalist.
- [g4media.ro] *8w* `probabil_curat` (scor=0):
  > ”Cred că ameninţarea războiului nuclear este o cacealma.
- [g4media.ro] *7w* `probabil_curat` (scor=0):
  > Primesc informaţii şi de la alte naţiuni.
- [digi24.ro] *13w* `probabil_curat` (scor=0):
  > „În cameră, găsești documentarele noastre despre realitatea războiului din Ucraina”, a explicat Mukka.
- [g4media.ro] *10w* `probabil_curat` (scor=0):
  > Facem totul pentru a-i ajuta pe băieții noștri de acolo.
- [digi24.ro] *9w* `probabil_curat` (scor=0):
  > Amal Clooney condamnă crimele de război comise de Rusia.
- [digi24.ro] *12w* `probabil_curat` (scor=0):
  > „Un rânjet al «lumii barbare ruse», pentru care nimic nu e sfânt.
- [g4media.ro] *7w* `probabil_curat` (scor=0):
  > ONU cere încetarea „suferinţei îngrozitoare” în Ucraina.

### medii (15-34 cuvinte)

- [g4media.ro] *18w* `probabil_curat` (scor=0):
  > O victorie militară ucraineană, expulzându-i pe ruşi din toată Ucraina, inclusiv Crimeea, are o probabiliate nu foarte ridicată.
- [g4media.ro] *24w* `probabil_curat` (scor=0):
  > Franţa a anunţat la sfârşitul lunii ianuarie că intenţionează să trimită „mai multe sute” de soldaţi în România, în cadrul unei posibile dislocări NATO.
- [digi24.ro] *21w* `probabil_curat` (scor=0):
  > În orașul Melitopol, în sud-estul Ucrainei, loclanicii nu s-au retras nici când soldații ruși au tras focuri de avertisment în aer.
- [g4media.ro] *23w* `probabil_curat` (scor=0):
  > Find out more about Defence Intelligence's use of language: https://t.co/NrXBzGy3WF 🇺🇦 #StandWithUkraine 🇺🇦 pic.twitter.com/nLJmQiKznq — Ministry of Defence 🇬🇧 (@DefenceHQ) July 4, 2023
- [digi24.ro] *29w* `probabil_curat` (scor=0):
  > În acelai timp, un oficial NATO, citat de presa americană sub protecția anonimatului, a anunțat ca fiind estimat între 7.000 și 15.000 de morți bilanțul în rândul armatei ruse.
- [g4media.ro] *17w* `probabil_curat` (scor=1):
  > WSJ: Casa Albă cere Congresului american peste 100 de miliarde de dolari pentru Israel, Ucraina și palestinieni.
- [digi24.ro] *25w* `probabil_curat` (scor=0):
  > „Rapoarte îngrijorătoare: Rușii ar fi îndreptat mai multe sisteme de lansare de rachete din satul rusesc de frontieră Popovka spre propriul teritoriu”, a scris Kuleba.
- [g4media.ro] *20w* `probabil_curat` (scor=0):
  > Rusia a încercat să desfăşoare acest tip de atac în trecut, însă el rămâne ”rar”, declară AFP purtătorul de cuvânt.
- [g4media.ro] *17w* `probabil_curat` (scor=0):
  > Această decizie vine în urma unei serii de lovituri rusești asupra infrastructurii energetice ucrainene legate de Baku.
- [g4media.ro] *19w* `probabil_curat` (scor=0):
  > Țara noastră a cheltuit până acum 51 milioane de lei, iar autoritățile încep decontările la Bruxelles, a anunțat Ciucă.
- [digi24.ro] *29w* `probabil_curat` (scor=0):
  > Lapin este considerat responsabil pentru înfrângerea rușilor pe frontul de la Liman și a fost criticat aspru de liderul cecen Ramzan Kadîrov și de șeful grupului Wagner, Evgheni Progojin.
- [g4media.ro] *22w* `probabil_curat` (scor=0):
  > Această inspecţie intervine în timp ce atacuri intense ale trupelor ruse vizează în ultimele săptămâni platforma industrială din Avdiivka, în estul Ucrainei.
- [g4media.ro] *28w* `probabil_curat` (scor=0):
  > Guvernul face un apel la cetățeni, autorități publice locale, agenți economici de a reduce consumul de energie, în special în orele de vârf, între 06:00-09:00 și între 17:00-23:00.
- [digi24.ro] *15w* `probabil_curat` (scor=0):
  > Este vorba despre un nou model de submarin denumit „Kronos”, după personajul din mitologia greacă.
- [g4media.ro] *32w* `probabil_curat` (scor=0):
  > Vedem acţiuni menite să le limiteze în mod deliberat propria suveranitate şi aşa mai departe”, a spus Peskov, arătând că este de acord cu vicepreşedintele Consiliului de Securitate al Rusiei, Dmitri Medvedev.

### lungi (35-59 cuvinte)

- [digi24.ro] *47w* `probabil_curat` (scor=0):
  > Însă această cotă de popularitate a scăzut la 77% în decembrie anul trecut, la 64% în februarie şi la 59% în luna mai, a transmis acest institut într-un comunicat, citând sondajul său realizat la mijlocul lui mai pe un eşantion reprezentativ de peste o mie de persoane.
- [g4media.ro] *58w* `probabil_curat` (scor=1):
  > Armata ucraineană a afirmat vineri că a avut „succese în două zone ale frontului din sud”, relatează CNN . „În direcțiile Novodanylivka – Robotyne și Mala Tokmachka – Novofedorivka, au obținut un succes parțial și se înfig în frontierele obținute”, a declarat Andriy Kovalov, un purtător de cuvânt al
- [g4media.ro] *44w* `probabil_curat` (scor=0):
  > Occidentul ar trebui să înăsprească sancţiunile împotriva Rusiei şi să livreze Kievului rachete cu rază lungă de acţiune, ca răspuns la cele mai recente bombardamente ale Moscovei asupra Ucrainei, a declarat miercuri ministrul polonez al afacerilor externe Radoslaw Sikorski, informează Reuters, cita
- [g4media.ro] *36w* `probabil_curat` (scor=0):
  > El a precizat că preşedintele american şi omologul său ucrainean, Volodimir Zelenski, vor avea joi o reuniune bilaterală urmată de o conferinţă de presă comună, în luxoasa staţiune de pe litoral Borgo Egnazia, din Apulia (sud).
- [g4media.ro] *42w* `probabil_curat` (scor=0):
  > „În baza datelor comunicate de SA Energocom, pentru ziua de vineri, 27 martie 2026, se înregistrează un deficit semnificativ de energie electrică în intervalul orelor de vârf (18:00-22:00), generat de insuficiența capacităților de import disponibile”, a transmis vineri Guvernul de la Chișinău.
- [g4media.ro] *47w* `probabil_curat` (scor=0):
  > Prima astfel de ieșire a lui Putin de la invazia Ucrainei ar fi avut loc miercuri, chiar de Kurban Bayram, sărbătoarea sacrificiului în Islam, și la câteva zile după ce Evgheni Prigojin ar fi fost exilat în Belarus în urma presupusei revolte și a marșului spre Moscova.
- [g4media.ro] *40w* `probabil_curat` (scor=0):
  > ”Trebuie să repunem drapelul ucrainean în toate oraşele şi comunităţile din Ucraina, trebuie să ne asigurăm de responsabilizarea reală a statului terorist pentru acest război şi trebuie să garantăm siguranţa tuturor generaţiilor de ucraineni după sfârşitul acestui război”, subliniază el.
- [digi24.ro] *37w* `probabil_curat` (scor=0):
  > Şeful statului a adăugat că trebuie schimbată și comunicarea privid Ucraina, în sensul că trebuie explicat românilor că „ securitatea Ucrainei şi ajutorul românesc pentru Ucraina țin de securitatea României şi de cea a Republicii Moldova ”.
- [g4media.ro] *36w* `probabil_curat` (scor=0):
  > O societate elenă vrea să încheie un acord pe 20 de ani pentru GNL americane / România este printre potenţialii cumpărători / Europa se pregătește de oprirea importurilor de gaze rusești până la finalul lui 2027.
- [g4media.ro] *38w* `probabil_curat` (scor=0):
  > Polonia intenţionează ”să crească producţia de cărbune termic din minele existente în acest an cu maximum 1,5 milioane de tone”, a declarat Janusz Olszowski, preşedintele Camerei de Industrie şi Comerţ Minieră din Polonia, într-o declaraţie trimisă prin e-mail.
- [digi24.ro] *41w* `probabil_curat` (scor=0):
  > Şi astăzi am avut o dezbatere foarte puternică cu el la Bruxelles, fiindcă indiferent cât de tare ţipă cineva la mine sau indiferent cât de agresiv este cineva, eu tot de partea păcii sunt şi noi, maghiarii, suntem de partea păcii.
- [g4media.ro] *48w* `probabil_curat` (scor=0):
  > Viceministrul ucrainean al apărării, Hanna Maliar, a declarat luni că forţele Kievului continuă să controleze o parte din sud-vestul oraşului Bahmut, dezminţind din nou afirmaţiile şefului grupului de mercenari ruşi Wagner, Evgheni Prigojin, potrivit cărora luptătorii săi ar fi preluat controlul înt
- [digi24.ro] *53w* `probabil_curat` (scor=0):
  > În ciuda întâlnirilor lui Trump cu președintele rus Vladimir Putin în Alaska, pe 15 august, și cu Volodimir Zelenski și liderii europeni, pe 18 august, se pare că s-au înregistrat puține progrese în ceea ce privește încetarea ostilităților sau organizarea unei întâlniri față în față între Zelenski ș
- [g4media.ro] *46w* `probabil_curat` (scor=0):
  > Preşedintele Rumen Radev a respins acordul ratificat anterior între Ministerul bulgar de Interne şi Ministerul ucrainean al Apărării privind furnizarea de vehicule blindate de transport Ucrainei şi l-a returnat Parlamentului pentru o nouă discuţie, a declarat luni biroul de presă prezidenţial, infor
- [g4media.ro] *39w* `probabil_curat` (scor=0):
  > Ora 18:09 Intensificarea atacurilor cu rachete ale Rusiei în Ucraina are rolul, parţial, să epuizeze capacitatea de apărare antiaeriană a Kievului şi să obţină în sfârşit dominaţia cerului deasupra acestei ţării, a declarat sâmbătă un înalt oficial al Pentagonului.

### foarte_lungi (>=60 cuvinte)

- [digi24.ro] *73w* `probabil_curat` (scor=0):
  > Leonid Manakov, şeful aşa-zisei reprezentanţe a Transnistriei la Moscova, nu are nicio atribuţie în legătură cu Acordul din 21 iulie 1992 care stipulează statutul forţei de menţinere a păcii în Transnistria, aşa că nu este în măsură să se exprime cu privire la mărirea numărului de „pacificatori” ruş
- [g4media.ro] *94w* `probabil_curat` (scor=0):
  > Rheinmetall va deschide o fabrică de vehicule blindate în Ucraina în următoarele 12 săptămâni, înlăturând astfel îngrijorările pe care alte companii occidentale din domeniul apărării le-ar avea cu privire la construirea unei prezențe în această țară în timp ce aceasta se află în război cu Rusia, rel
- [digi24.ro] *64w* `probabil_curat` (scor=1):
  > Armata ucraineană pierde în prezent între 40 și 45 de drone de recunoaștere în fiecare zi de conflict, a declarat șeful Serviciului de Comunicații Speciale al Ucrainei, Iurii Șîhol, informează CNN . Conform lui Șîhol, forțele armate ucrainene pierd tot felul de drone, de la „cele de bază”, precum Ma
- [digi24.ro] *62w* `probabil_curat` (scor=1):
  > Siegfrid Mureșan: „Să stăm departe de acest virus” Scumpiri în lanț în București: biletul STB ar putea ajunge la 5 lei, după ce și Metrorex a anunțat majorări Avertismentul lansat de JD Vance pentru Iran, înainte de discuțiile de la Islamabad: „Așteptăm cu interes negocierile” Accident grav în Sibiu
- [g4media.ro] *66w* `probabil_curat` (scor=1):
  > Kievul și Zagreb au convenit asupra „posibilității” de a folosi porturile croate de pe Dunăre pentru a exporta cereale ucrainene, a declarat ministrul de externe Dmytro Kuleba după o întâlnire cu omologul său Gordan Grlic-Radman, relatează CNN . „Am convenit asupra posibilității de a utiliza porturi
- [g4media.ro] *60w* `probabil_curat` (scor=0):
  > Doi dintre cei mai fervenți susținători ai lui Putin se sperie din cauza eforturilor dezastruoase de mobilizare în masă, de care se tem că ar putea duce la o revoltă majoră împotriva războiului din Ucraina, notează dailybeast.com . „Mobilizarea parțială” a Rusiei a aruncat o altă umbră asupra situaț
- [digi24.ro] *63w* `probabil_curat` (scor=1):
  > El a menţionat că România contribuie astfel la asigurarea expertizei necesare personalului ucrainean pentru operarea, întreţinerea şi mentenanţa aeronavelor de luptă F-16, conform angajamentului asumat în Declaraţia comună semnată în marja Summitului NATO de la Vilnius de către miniştrii apărării di
- [digi24.ro] *62w* `probabil_curat` (scor=0):
  > Încercările Rusiei de a stârni un conflict în această regiune de graniță a Ucrainei au eșuat însă, iar sentimentul prorus „s-a prăbușit” complet după invazia Rusiei, relatează The Economist . Forțele ucrainene i-au oprit pe rușii care au vrut să trimită în Bugeac trupe de comando la începutul invazi
- [digi24.ro] *79w* `probabil_curat` (scor=1):
  > Pugaciova a spus că invazia rusească este o „povară pentru oamenii de rând” și duce la „moartea băieților noștri în scopuri iluzorii”, relatează Reuters . În vârstă de 73 de ani, Alla Pugaciova, o celebritate și în perioada sovietică şi în cea post-sovietică, a cerut, de asemenea, să fie clasificată
- [digi24.ro] *67w* `probabil_curat` (scor=0):
  > Zelenski a sosit la Damasc însoţit de ministrul turc de externe Hakan Fidan, amândoi venind de la Istanbul, unde preşedintele ucrainean s-a întâlnit sâmbătă cu omologul său turc Recep Tayyip Erdogan pentru discuţii despre eforturile de a pune capăt războiului cu Rusia, precum şi despre securitatea e
- [digi24.ro] *61w* `probabil_curat` (scor=0):
  > Experimentatul politician de 60 de ani promite să pună capăt „greșelii fatale” pe care Putin a făcut-o în Ucraina, să oprească mobilizarea și să elibereze prizonierii politici, inclusiv pe Alexei Navalnîi, relatează Politico . În ultimele săptămâni, cozi lungi de oameni s-au format în preajma birour
- [g4media.ro] *66w* `probabil_curat` (scor=0):
  > Preşedintele ucrainean Volodimir Zelenski a declarat vineri că hidrocentrala Kaniv, de la sud-est de Kiev, şi o alta aflată pe Nistru au fost atacate în noul raid aerian rusesc cu rachete şi drone desfăşurat în cursul nopţii asupra infrastructurii energetice ucrainene şi care, potrivit operatorului 
- [g4media.ro] *65w* `probabil_curat` (scor=0):
  > Liderii Franței, Italiei, Germaniei, Marii Britanii, Finlandei, Poloniei, Portugaliei și șefa Comisiei Europene au mai transmis că ”Ucraina trebuie să aibă garanții de securitate de neclintit pentru a-și apăra în mod eficient suveranitatea și integritatea teritorială”, dar și că ”Ucraina poate conta
- [digi24.ro] *64w* `probabil_curat` (scor=1):
  > Dată fiind proximitatea R. Moldova de Ucraina şi de Rusia, Maia Sandu a declarat că ţara sa se află în „prima linie” a războiului hibrid al Moscovei împotriva Occidentului, subliniind că Republica Moldova luptă în prezent pentru „apărarea democraţiei împotriva atacurilor cibernetice, dezinformării” 
- [g4media.ro] *71w* `probabil_curat` (scor=0):
  > Dar, într-o altă tiradă plină de înjurături, publicată în aceeași zi în care Kremlinul a comemorat victoria asupra Germaniei naziste, Prigojin a declarat că o brigadă rusă și-a abandonat pozițiile la sud de Bahmut, ceea ce a dus la numeroase pierderi în rândul luptătorilor săi, relatează CNN . În co

## 6. Sumar decizional

- Propoziții `probabil_curat`: **5,944** (99.46%)
- Propoziții detectate ca zgomot: **32** (0.54%)

**Pași următori:**
1. Citește eșantionul manual (Secțiunea 5) și marchează FP/FN observate.
2. Ajustează regulile (lărgire/restrângere pattern, prag scor_suspect) pe baza FP/FN.
3. Abia apoi rulează scriptul de filtrare finală + deduplicare + embeddings.