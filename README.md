# 🏥 PrevIA — Assistente de Medicina Preventiva

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://langchain-ai.github.io/langgraph/)

Sistema agêntico de medicina preventiva baseado em RAG que consulta diretrizes médicas públicas (Ministério da Saúde, INCA, OMS) para responder perguntas sobre prevenção e gerar planos de saúde personalizados com citações das fontes e mecanismo de verificação anti-alucinação.

> ⚠️ Este sistema fornece informações educativas baseadas em diretrizes públicas. **Não substitui consulta com profissional de saúde.**

---

## 🎯 Problema

Muitas pessoas não sabem quais exames preventivos fazer nem quando fazê-los. Informações na internet são confusas e de fontes não confiáveis, causando diagnósticos tardios, exames desnecessários e falta de prevenção.

## 💡 Solução

O PrevIA é um assistente baseado em evidências que:
- Responde perguntas sobre prevenção com citações de fontes oficiais
- Gera planos preventivos anuais personalizados por perfil (idade, sexo, histórico)
- Verifica automaticamente se as respostas têm suporte nos documentos (anti-alucinação)
- Integra um servidor MCP próprio (`health-checklist`) para geração de checklists

---

## 🤖 Arquitetura dos Agentes (LangGraph)

```
Usuário
   ↓
Supervisor Agent → decide rota: Q&A ou Automação
   ↓  (Q&A)                        ↓ (Automação)
Retriever Agent            Automation Agent
   ↓                         (MCP health-checklist)
Writer Agent                       ↓
   ↓                           Safety Agent
Self-Check ──re-busca           ↓
   ↓ aprovado              Resposta Final
Safety Agent
   ↓
Resposta Final
```

| Agente | Função |
|--------|--------|
| **Supervisor** | Classifica intenção: Q&A ou geração de plano |
| **Retriever** | Busca FAISS com embedding português (k=8) |
| **Writer** | Gera resposta com citações obrigatórias |
| **Self-Check** | Valida se resposta tem suporte nos documentos |
| **Safety** | Adiciona disclaimer médico |
| **Automation** | Gera plano preventivo anual via MCP |

---

## 🗂️ Corpus RAG — 38 documentos públicos

| Categoria | Documentos |
|-----------|-----------|
| Rastreamento e câncer | 9 |
| Doenças crônicas e atenção básica | 10 |
| Saúde da mulher e gestação | 4 |
| Vacinação (calendários nacionais) | 5 |
| Nutrição e estilo de vida | 5 |
| Saúde bucal, emocional e outros | 4 |
| **Total** | **39** |

Fontes: Ministério da Saúde, INCA, OMS — todos documentos públicos.

---

## 🔧 MCP — Model Context Protocol

### Servidor: `health-checklist` (implementação própria)

**Tool exposta:** `generate_preventive_checklist`

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `age` | integer | Idade do paciente |
| `sex` | string | Sexo (masculino/feminino) |
| `risk_factors` | string | Fatores de risco separados por vírgula |

### Controles de segurança

| Controle | Implementação |
|----------|--------------|
| **Allowlist** | Apenas 1 tool permitida (`generate_preventive_checklist`) |
| **Sem acesso externo** | Sem acesso a disco do sistema, internet ou execução de código |
| **Logs de auditoria** | Toda chamada registrada em `mcp_audit.log` com timestamp |
| **Bloqueio** | Chamadas a tools fora da allowlist são bloqueadas e registradas |

**O que o agente NÃO pode fazer via MCP:**
- Acessar arquivos do sistema além do corpus indexado
- Fazer requisições HTTP externas
- Executar comandos shell
- Chamar qualquer tool além de `generate_preventive_checklist`

**Justificativa da escolha:** MCP próprio garante controle total sobre o escopo da ferramenta e elimina riscos de supply-chain (exfiltração de dados, execução arbitrária) comuns em servidores MCP de terceiros.

---

## 📊 Avaliação
 
### RAG — 20 perguntas rotuladas (metodologia RAGAS)
 
| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Faithfulness | 0.740 | Respostas fiéis ao contexto na maioria dos casos |
| Answer Relevancy | 0.750 | Respostas altamente relevantes às perguntas |
| Context Precision | 0.677 | Retriever encontra documentos relevantes |
| Context Recall | 0.600 | Corpus cobre bem as respostas esperadas |
| **Latência média** | **11.7s** | Modelo local sem GPU dedicada |
 
15 de 20 perguntas com score ≥ 0.7 em Answer Relevancy. 
 
> Métricas calculadas com Qwen2.5 7B + embedding `BAAI/bge-small-en-v1.5` seguindo metodologia RAGAS.

### Automação MCP — 5 tarefas
 
| Perfil testado | Resultado |
|---------------|-----------|
| Feminino, 35a, diabetes na família | ✅ |
| Feminino, 50a, câncer de mama na família | ✅ |
| Masculino, 45a, hipertensão + sedentário | ✅ |
| Feminino, 28a, sem histórico | ✅ |
| Masculino, 60a, tabagista + diabetes | ✅ |
 
| Métrica | Valor |
|---------|-------|
| Taxa de sucesso | 100% (5/5) |
| Nº médio de steps | 4 (MCP + retriever + prompt + LLM) |
| Docs consultados por tarefa | 15 |
| Latência média | 54.1s por tarefa |
 
> Latência elevada esperada — modelo 7B local sem GPU dedicada processando 15 documentos por tarefa.
 
### MCP
 
| Item | Detalhe |
|------|---------|
| Servidor | `health-checklist` |
| Tool | `generate_preventive_checklist` |
| Allowlist | 1 tool |
| Logs | `mcp_audit.log` — timestamp + parâmetros + resultado |
| Bloqueios | Tools não autorizadas registradas e recusadas |
 
---

## ⚙️ Stack

| Componente | Tecnologia |
|------------|-----------|
| LLM | Qwen2.5 7B via Ollama (local) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Busca vetorial | FAISS |
| Orquestração | LangGraph + LangChain |
| Interface | Streamlit + ngrok |
| MCP | health-checklist (implementação própria) |
| Avaliação | Metodologia RAGAS (implementação local) |
| Ambiente | Google Colab + GPU T4 |

---

## 🚀 Como executar

### Pré-requisitos
- Google Colab com GPU T4 ativada
- Token do ngrok (gratuito em [dashboard.ngrok.com](https://dashboard.ngrok.com))
- Pasta pública no Google Drive com os PDFs do corpus

### Execução

1. Abra o `PrevIA.ipynb` no Google Colab
2. Ative a GPU: **Editar → Configurações do notebook → GPU T4**
3. Execute todas as células em ordem (`Runtime → Run all`)
4. Acesse a URL gerada pelo ngrok

---

## 📁 Estrutura do repositório

```
previa/
├── PrevIA.ipynb              # Notebook principal (único arquivo para executar)
├── app/
│   └── streamlit_app.py      # Interface Streamlit
├── src/
│   └── mcp/
│       └── health_checklist_server.py  # Servidor MCP
├── eval/
│   ├── resumo_metricas.json  # Métricas RAG
│   ├── resumo_automacao.json # Métricas automação
│   └── ragas_resultados.csv  # Detalhamento por pergunta
├── docs/
│   └── corpus_info.md        # Documentação do corpus
├── Dockerfile                # Container para deploy
├── LICENSE                   # MIT
├── CITATION.cff              # Citação acadêmica
└── README.md
```

---

## ⚠️ Limitações

- Respostas baseadas exclusivamente nos documentos indexados
- Modelo 7B local ocasionalmente ignora contexto em perguntas complexas
- Corpus não inclui diretrizes publicadas após a data dos documentos indexados
- Não considera histórico clínico completo do paciente
- **Não substitui consulta médica em nenhuma hipótese**

## 🔮 Próximos passos

- Ampliar corpus com mais diretrizes
- Implementar reranking dos documentos recuperados
- Avaliar modelos maiores (13B+) para maior fidelidade
- Deploy permanente com Docker + cloud
- MCP de nutrição

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE)

## 📚 Citação

```bibtex
@software{previa2025,
  title  = {PrevIA: Assistente de Medicina Preventiva},
  year   = {2025},
  url    = {https://github.com/leilafarias/previa}
}
```
