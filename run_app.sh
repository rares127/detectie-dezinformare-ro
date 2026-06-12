#!/usr/bin/env bash
# Script convenience pentru rularea aplicatiei FastAPI.
# Foloseste din radacina proiectului: ./run_app.sh

set -e

# Verificam ca suntem in radacina proiectului
if [ ! -d "app" ] || [ ! -d "models" ]; then
    echo "EROARE: rulează acest script din rădăcina proiectului (folder cu app/ și models/)."
    exit 1
fi

echo "🚀 Pornesc detectorul de dezinformare..."
echo "   URL: http://127.0.0.1:8000"
echo "   Press Ctrl+C pentru a opri."
echo ""

# NU folosim --reload (modelele se reincarca lent)
uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level info
