import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# Lista ta perfecta de URL-uri
urls = [
    "https://stirileprotv.ro/stiri/international/putin-sustine-ca-miza-conflictului-din-ucraina-este-existenta-rusiei-ca-stat-este-o-misiune-pentru-supravietuirea-rusiei.html",
    "https://stirileprotv.ro/stiri/international/ucraina-acuza-moscova-ca-vrea-sa-destabilizeze-transnistria-zmoldova-sa-se-pregateasca-sa-primeasca-oaspeti.html",
    "https://stirileprotv.ro/stiri/international/inca-o-noapte-de-groaza-in-ucraina-rachetele-si-dronele-moscovei-au-tintit-civilii-din-orasul-ternopil-sunt-zeci-de-victime.html",
    "https://stirileprotv.ro/stiri/international/sefa-diplomatiei-europene-mesaj-pentru-donald-trump-europa-nu-poate-sprijini-singura-ucraina-in-razboiul-cu-rusia.html",
    "https://stirileprotv.ro/stiri/international/peste-78-000-de-soldati-rusi-au-murit-pe-front-rata-de-crestere-a-deceselor-este-un-record-de-la-inceputul-razboiului.html",
    "https://hotnews.ro/rusia-cere-nato-sa-abandoneze-extinderea-spre-est-si-sa-anuleze-decizia-cheie-luata-la-summitul-de-la-bucuresti-2173631",
    "https://hotnews.ro/ucraina-blocheaza-active-de-464-milioane-de-dolari-ale-unor-oligarhi-rusi-40026",
    "https://hotnews.ro/un-general-rus-avertizeaza-ca-tancurile-leopard-sunt-cele-mai-bune-din-europa-vor-crea-o-amenintare-foarte-serioasa-84683",
    "https://hotnews.ro/video-baie-de-snge-peste-100-de-soldati-rusi-ar-fi-fost-ucisi-la-soledar-joi-pentru-fiecare-100-de-metri-rusia-plateste-cu-50-de-morti-86946",
    "https://hotnews.ro/zelenski-apel-catre-aliati-pe-fondul-noilor-discutii-cu-sua-flota-din-umbra-a-rusiei-nu-trebuie-sa-se-simta-in-siguranta-in-apele-europene-2200213",
    "https://www.libertatea.ro/stiri/drone-rusesti-aproape-romania-mesaj-ro-alert-tulcea-tinta-intrat-spatiu-aerian-5702631",
    "https://www.libertatea.ro/stiri/rusii-au-mai-sters-de-pe-fata-pamantului-un-oras-ucrainean-pe-care-l-bombardeaza-de-12-zile-imagini-ca-dupa-apocalipsa-in-vovceansk-4897447",
    "https://www.libertatea.ro/stiri/volodimir-zelenski-se-angajeaza-sa-recupereze-crimeea-4461215",
    "https://www.libertatea.ro/stiri/zelenski-capturare-prime-ora-invazie-rusia-4105030",
    "https://www.libertatea.ro/stiri/ucraina-a-folosit-deja-arme-occidentale-pentru-atacuri-pe-teritoriul-rusiei-scrie-afp-putin-a-amenintat-cu-consecinte-grave-4904411"
]

rezultate = []
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

print("Incepem colectarea articolelor...")

for idx, url in enumerate(urls):
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 1. Extragere Titlu
        titlu_tag = soup.find('h1')
        titlu = titlu_tag.text.strip() if titlu_tag else soup.title.text.strip()
        
        # 2. Setare Sursa
        if "stirileprotv.ro" in url: sursa = "stirileprotv.ro"
        elif "hotnews.ro" in url: sursa = "hotnews.ro"
        elif "libertatea.ro" in url: sursa = "libertatea.ro"
        else: sursa = "necunoscuta"
        
        # 3. Extragere Text (luam paragrafele relevante)
        paragrafe = soup.find_all('p')
        # Pastram doar <p>-urile care au peste 60 de caractere pentru a fenta reclamele/linkurile interne
        text_bucati = [p.text.strip() for p in paragrafe if len(p.text.strip()) > 60]
        text_complet = " ".join(text_bucati)
        
        # Curatam textul pentru a nu strica formatul CSV
        text_complet = text_complet.replace('\n', ' ').replace('\r', '').replace('"', '„')
        titlu = titlu.replace('"', '„')
        
        # 4. Extragere An brut
        match_an = re.search(r'\b(202[2-6])\b', resp.text)
        an = int(match_an.group(1)) if match_an else 2024
        
        rezultate.append({
            'id': f'ext_{idx+1:03d}',
            'url': url,
            'titlu': titlu,
            'data': f"{an}-01-01", # Data exacta e irelevanta pentru similaritatea semantica
            'an': an,
            'sursa_site': sursa,
            'stire_citata': text_complet,
            'label_numeric': 0
        })
        print(f"✅ Descărcat: [{sursa}] {titlu[:40]}...")
        
    except Exception as e:
        print(f"❌ Eroare la {url}: {e}")

# Salvare CSV
df = pd.DataFrame(rezultate)
# Cream folderul data/raw daca nu exista
import os
os.makedirs('data/raw', exist_ok=True)

output_path = "data/raw/test_cls0_external.csv"
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n🚀 GATA! Am salvat {len(df)} articole perfect formatate în {output_path}")