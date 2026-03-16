#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   PrevIA — Assistente de Medicina Preventiva ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Inicia Ollama em background ─────────────────────────
echo "Iniciando Ollama..."
ollama serve &
OLLAMA_PID=$!

# Aguarda Ollama responder
for i in $(seq 1 30); do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "Ollama pronto!"
        break
    fi
    sleep 2
done

# ── 2. Baixa modelo se necessário ──────────────────────────
if ! ollama list 2>/dev/null | grep -q "qwen2.5:7b"; then
    echo "⏳ Baixando Qwen2.5 7B (~5GB, apenas na primeira execução)..."
    ollama pull qwen2.5:7b
    echo "Modelo baixado!"
else
    echo "Modelo Qwen2.5 7B já disponível!"
fi

# ── 3. Indexa o corpus se necessário ───────────────────────
if [ ! -f "previa/faiss_db/index.faiss" ]; then
    echo "⏳ Indexando corpus (apenas na primeira execução)..."
    python setup.py --only-index
    echo "Corpus indexado!"
else
    echo "Índice FAISS já existe!"
fi

# ── 4. Inicia Streamlit ─────────────────────────────────────
echo ""
echo "PrevIA rodando em http://localhost:8501"
echo ""

streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
