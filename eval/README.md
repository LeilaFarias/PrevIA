# Avaliação — PrevIA

## Arquivos

- `resumo_metricas.json` — métricas RAG resumidas (gerado pelo notebook)
- `resumo_automacao.json` — métricas de automação MCP (gerado pelo notebook)
- `ragas_resultados.csv` — detalhamento por pergunta (gerado pelo notebook)
- `respostas_sistema.json` — respostas completas do sistema (gerado pelo notebook)

## Como gerar

Execute a seção **Avaliação** do `PrevIA.ipynb`. Os arquivos serão salvos
automaticamente em `/content/previa/eval/` e podem ser baixados do Colab.

## Metodologia

Métricas calculadas com Qwen2.5 7B local seguindo a metodologia RAGAS:
- **Faithfulness** — fidelidade da resposta ao contexto recuperado
- **Answer Relevancy** — relevância da resposta à pergunta
- **Context Precision** — precisão dos documentos recuperados
- **Context Recall** — cobertura dos documentos sobre a resposta esperada

