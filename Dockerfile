FROM python:3.12-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    curl wget git \
    poppler-utils \
    tesseract-ocr tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# Instala Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o projeto
COPY . .

# Cria pastas necessárias
RUN mkdir -p previa/docs previa/faiss_db previa/eval previa/src/mcp

# Porta do Streamlit
EXPOSE 8501

# Script de entrada
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
