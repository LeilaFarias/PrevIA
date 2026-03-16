import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain_core.documents import Document
import logging, os

# ── Logger MCP ─────────────────────────────────────────────
logger = logging.getLogger("mcp_audit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("/content/previa/mcp_audit.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)

# ── Recursos cached ─────────────────────────────────────────
@st.cache_resource
def carregar_recursos():
    embeddings = HuggingFaceEmbeddings(
        model_name="neuralmind/bert-base-portuguese-cased",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        "/content/previa/faiss_db", embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)
    return retriever, llm

retriever, llm = carregar_recursos()

# ── MCP Tool ────────────────────────────────────────────────
@tool
def generate_preventive_checklist(age: int, sex: str, risk_factors: str = "nenhum") -> str:
    """[MCP: health-checklist] Gera checklist preventivo personalizado."""
    logger.info(f"CHAMADA | tool=generate_preventive_checklist | age={age} sex={sex}")
    docs = retriever.invoke(f"exames preventivos {sex} {age} anos {risk_factors}")
    contexto, fontes = "", []
    for doc in docs:
        src = doc.metadata.get("source_file", "?")
        pag = doc.metadata.get("page", "?")
        contexto += f"[{src}, pag.{pag}]\n{doc.page_content}\n\n"
        fontes.append(f"{src} pag.{pag}")
    prompt = ChatPromptTemplate.from_template("""
Gere checklist preventivo para:
- Idade: {age} anos | Sexo: {sex} | Fatores de risco: {risk_factors}
Use APENAS os trechos. Nao invente exames ou vacinas.
TRECHOS: {contexto}
EXAMES: [apenas os que aparecem nos trechos]
VACINAS: [apenas as que aparecem nos trechos]
HABITOS: [apenas os que aparecem nos trechos]
FONTES: {fontes}""")
    result = (prompt | llm).invoke({"age": age, "sex": sex,
        "risk_factors": risk_factors, "contexto": contexto, "fontes": ", ".join(fontes)})
    logger.info(f"SUCESSO | tool=generate_preventive_checklist | chars={len(result.content)}")
    return result.content

# ── State ───────────────────────────────────────────────────
DISCLAIMER = "\n\n---\n> ⚠️ **Aviso:** Informações educativas baseadas em diretrizes públicas do Ministério da Saúde e OMS. Não substitui consulta com profissional de saúde."

class PrevIAState(TypedDict):
    pergunta: str
    perfil: Optional[dict]
    rota: str
    documentos: List[Document]
    resposta: str
    check_ok: bool
    tentativas: int
    resposta_final: str

PROMPT_WRITER = ChatPromptTemplate.from_template("""
Você é o PrevIA, assistente de medicina preventiva brasileiro.
PERGUNTA: {pergunta}
TRECHOS: {contexto}
INSTRUCOES:
1. Leia todos os trechos com atenção
2. Extraia QUALQUER informação relevante, mesmo que parcial
3. Se encontrar idade, faixa etária, frequência — inclua na resposta
4. Responda em português com marcadores
5. Cite fontes: [Fonte: arquivo, pag. X]
6. Só diga nao encontrei se realmente nao houver nada relacionado
RESPOSTA:""")

PROMPT_CHECK = ChatPromptTemplate.from_template("""
Verifique se a RESPOSTA usa informações presentes nos TRECHOS.
TRECHOS: {contexto}
RESPOSTA: {resposta}
- APROVADO se menciona pelo menos 1 informação dos trechos
- APROVADO se diz que nao encontrou evidência
- REPROVADO apenas se CONTRADIZ os trechos
Responda APENAS: APROVADO ou REPROVADO""")

# ── Agentes ─────────────────────────────────────────────────
def supervisor(state):
    gatilhos = ["plano","planejamento","gerar plano","meu perfil","tenho anos","sou homem","sou mulher"]
    rota = "automacao" if any(g in state["pergunta"].lower() for g in gatilhos) else "qa"
    return {**state, "rota": rota, "tentativas": 0}

def retriever_agent(state):
    pergunta = state["pergunta"]
    if state.get("perfil"):
        p = state["perfil"]
        pergunta = f"{pergunta} idade {p.get('idade','')} sexo {p.get('sexo','')} {p.get('historico','')}"
    docs = retriever.invoke(pergunta)
    termos = {
        "mamografia": "mamografia rastreamento câncer mama faixa etária",
        "papanicolau": "papanicolau periodicidade frequência anos",
        "glicemia": "glicemia diabetes rastreamento jejum",
    }
    for termo, q in termos.items():
        if termo in state["pergunta"].lower():
            ids = {f"{d.metadata.get('source_file')}_{d.metadata.get('page')}" for d in docs}
            for d in retriever.invoke(q):
                chave = f"{d.metadata.get('source_file')}_{d.metadata.get('page')}"
                if chave not in ids:
                    docs.append(d); ids.add(chave)
            break
    return {**state, "documentos": docs}

def writer_agent(state):
    contexto = ""
    for i, doc in enumerate(state["documentos"]):
        src = doc.metadata.get("source_file","?")
        pag = doc.metadata.get("page","?")
        contexto += f"[Trecho {i+1} — {src}, pag.{pag}]\n{doc.page_content}\n\n"
    result = (PROMPT_WRITER | llm).invoke({"pergunta": state["pergunta"], "contexto": contexto})
    return {**state, "resposta": result.content}

def self_check(state):
    if not state["documentos"]:
        return {**state, "check_ok": True, "tentativas": state.get("tentativas",0)+1}
    contexto = "\n\n".join([d.page_content[:300] for d in state["documentos"]])
    result = (PROMPT_CHECK | llm).invoke({"contexto": contexto, "resposta": state["resposta"][:500]})
    ok = "REPROVADO" not in result.content.upper()
    return {**state, "check_ok": ok, "tentativas": state.get("tentativas",0)+1}

def rota_apos_check(state):
    if state["check_ok"]: return "aprovado"
    if state["tentativas"] < 2: return "re_busca"
    return "recusar"

def safety_agent(state):
    return {**state, "resposta_final": state["resposta"] + DISCLAIMER}

def recusar(state):
    return {**state, "resposta_final": "Não encontrei evidência suficiente para responder com segurança." + DISCLAIMER}

def selecionar_calendarios(perfil):
    idade = perfil.get("idade", 30)
    hist = perfil.get("historico","").lower() + perfil.get("condicoes","").lower()
    gestante = any(p in hist for p in ["gestante","grávida","gravida","gravidez"])
    if idade < 12: cals = ["Calendário Nacional de Vacinação - Criança.pdf"]
    elif idade < 20: cals = ["Calendário Nacional de Vacinação - Adolescentes e jovens.pdf"]
    elif idade >= 60: cals = ["Calendário Nacional de Vacinação - Idoso.pdf"]
    else: cals = ["Calendário Nacional de Vacinação - Adulto.pdf"]
    if gestante and "Gestante" not in cals[0]:
        cals.append("Calendário Nacional de Vacinação - Gestante.pdf")
    return cals

def automation_agent(state):
    perfil = state.get("perfil", {})
    calendarios = selecionar_calendarios(perfil)
    historico = perfil.get("historico", "nenhum")
    sexo = perfil.get("sexo", "")
    idade = perfil.get("idade", "")
    queries = [
        f"exames preventivos {sexo} {idade} anos rastreamento",
        f"hábitos saudáveis prevenção doenças atividade física alimentação",
        f"exames monitoramento {historico}",
    ] + calendarios
    docs_vistos, todos_docs = set(), []
    for query in queries:
        for doc in retriever.invoke(str(query).strip()):
            chave = f"{doc.metadata.get('source_file')}_{doc.metadata.get('page')}"
            if chave not in docs_vistos:
                docs_vistos.add(chave); todos_docs.append(doc)
        if len(todos_docs) >= 15: break
    contexto = ""
    for doc in todos_docs:
        src = doc.metadata.get("source_file","?"); pag = doc.metadata.get("page","?")
        contexto += f"[{src}, pag.{pag}]\n{doc.page_content}\n\n"
    perfil_str = "\n".join([f"- {k}: {v}" for k,v in perfil.items()])
    cal_str = " e ".join(calendarios)
    PROMPT_AUTO = ChatPromptTemplate.from_template("""
Você é o PrevIA, assistente de medicina preventiva brasileiro.
Gere Plano Preventivo Anual para:
{perfil}
ATENCAO: historico de {historico}. Inclua exames específicos para esse fator de risco.
TRECHOS: {contexto}
## Plano Preventivo Anual Personalizado
### Exames Recomendados [exames gerais + específicos para {historico}]
### Vacinas [do calendário {cal_str}]
### Hábitos Preventivos [dos trechos, priorizando {historico}]
### Cronograma Sugerido [por trimestre com meses]
### Fontes [nome_arquivo.pdf, pag. X]
""")
    result = (PROMPT_AUTO | llm).invoke({
        "perfil": perfil_str, "contexto": contexto,
        "cal_str": cal_str, "historico": historico
    })
    return {**state, "resposta": result.content, "check_ok": True, "documentos": todos_docs}

# ── Grafo ───────────────────────────────────────────────────
@st.cache_resource
def criar_grafo():
    g = StateGraph(PrevIAState)
    g.add_node("supervisor",       supervisor)
    g.add_node("retriever",        retriever_agent)
    g.add_node("writer",           writer_agent)
    g.add_node("self_check",       self_check)
    g.add_node("safety",           safety_agent)
    g.add_node("recusar",          recusar)
    g.add_node("automation_agent", automation_agent)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["rota"],
        {"qa": "retriever", "automacao": "automation_agent"})
    g.add_edge("retriever",        "writer")
    g.add_edge("writer",           "self_check")
    g.add_conditional_edges("self_check", rota_apos_check,
        {"aprovado": "safety", "re_busca": "retriever", "recusar": "recusar"})
    g.add_edge("automation_agent", "safety")
    g.add_edge("safety", END)
    g.add_edge("recusar", END)
    return g.compile()

app_graph = criar_grafo()

# ════════════════════════════════════════════════════════════
# UI — Dark theme clínico
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PrevIA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}

.stApp { background: #0d1117 !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(35,134,54,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: "Inter", sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3fb950;
    margin-bottom: 0.6rem;
}
.hero-titulo {
    font-family: "Syne", sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #f0f6fc;
    margin: 0;
    line-height: 1;
    letter-spacing: -1.5px;
}
.hero-titulo em {
    font-style: normal;
    color: #3fb950;
}
.hero-desc {
    color: #8b949e;
    font-size: 0.88rem;
    margin-top: 0.8rem;
    max-width: 520px;
    line-height: 1.6;
}
.hero-tags {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.tag {
    background: #161b22;
    border: 1px solid #30363d;
    color: #8b949e;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.tag.green {
    border-color: #238636;
    color: #3fb950;
    background: rgba(35,134,54,0.1);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-family: "Inter", sans-serif !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: #238636 !important;
    color: #f0f6fc !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: "Inter", sans-serif !important;
    font-size: 0.92rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #238636 !important;
    box-shadow: 0 0 0 3px rgba(35,134,54,0.2) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #484f58 !important;
}

/* ── Selectbox / Number ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}

/* ── Botões ── */
.stButton > button[kind="primary"] {
    background: #238636 !important;
    color: #f0f6fc !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: "Inter", sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: background 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2ea043 !important;
}

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
    font-size: 0.92rem;
    color: #e6edf3;
    line-height: 1.8;
}
.card.green-accent {
    border-left: 3px solid #238636;
}
.card.info {
    font-size: 0.83rem;
    color: #8b949e;
    line-height: 1.65;
    padding: 1rem 1.4rem;
}

/* ── Aviso ── */
.aviso {
    background: rgba(35,134,54,0.08);
    border: 1px solid #238636;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.8rem;
    color: #7ee787;
    margin-top: 1rem;
    line-height: 1.5;
}

/* ── Pills de fonte ── */
.fonte-pill {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #3fb950;
    font-weight: 500;
    margin: 3px 3px 0 0;
    font-family: "Inter", monospace;
}

/* ── Label fields ── */
.field-label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #484f58;
    margin-bottom: 4px;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #238636 !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary {
    color: #8b949e !important;
    font-size: 0.83rem !important;
}

/* ── Warning ── */
.stAlert {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #8b949e !important;
}

/* Labels dos widgets */
label, .stSelectbox label, .stNumberInput label {
    color: #8b949e !important;
    font-size: 0.83rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Assistente de medicina preventiva</div>
  <div class="hero-titulo">Prev<em>IA</em></div>
  <div class="hero-desc">
    Consulta 39 diretrizes públicas do Ministério da Saúde, INCA e OMS
    para responder perguntas sobre prevenção e gerar planos personalizados —
    com citações e verificação anti-alucinação.
  </div>
  <div class="hero-tags">
    <span class="tag green">RAG + LangGraph</span>
    <span class="tag green">MCP health-checklist</span>
    <span class="tag">Ministério da Saúde</span>
    <span class="tag">INCA</span>
    <span class="tag">OMS</span>
    <span class="tag">Qwen2.5 7B local</span>
    <span class="tag">39 documentos</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────
aba1, aba2 = st.tabs(["💬  Perguntas", "📋  Plano Preventivo"])

# ══════════════════════════════
# ABA 1 — Q&A
# ══════════════════════════════
with aba1:
    st.markdown("<br>", unsafe_allow_html=True)

    col_main, col_side = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown('<div class="field-label">Sua pergunta</div>', unsafe_allow_html=True)
        pergunta = st.text_input(
            "", placeholder="Ex: A partir de qual idade devo fazer mamografia?",
            label_visibility="collapsed", key="input_qa"
        )
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            perguntar = st.button("Perguntar →", type="primary", key="btn_qa", use_container_width=True)

    with col_side:
        st.markdown("""
        <div class="card info">
        <div style="color:#3fb950;font-size:0.72rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.7rem;">Pipeline</div>
        <div style="line-height:2;">
        <span style="color:#238636">①</span> Supervisor<br>
        <span style="color:#238636">②</span> Retriever<br>
        <span style="color:#238636">③</span> Writer<br>
        <span style="color:#238636">④</span> Self-check<br>
        <span style="color:#238636">⑤</span> Safety
        </div>
        </div>
        """, unsafe_allow_html=True)

    if perguntar:
        if pergunta.strip():
            with st.spinner("Consultando diretrizes..."):
                estado = {
                    "pergunta": pergunta, "perfil": None, "rota": "",
                    "documentos": [], "resposta": "", "check_ok": False,
                    "tentativas": 0, "resposta_final": ""
                }
                resultado = app_graph.invoke(estado)

            partes = resultado["resposta_final"].split("---")
            resposta_txt = partes[0].strip()
            aviso_txt    = partes[1].strip() if len(partes) > 1 else ""

            st.markdown(f'<div class="card green-accent">{resposta_txt}</div>', unsafe_allow_html=True)
            if aviso_txt:
                st.markdown(f'<div class="aviso">{aviso_txt}</div>', unsafe_allow_html=True)

            if resultado.get("documentos"):
                with st.expander("📄 Fontes consultadas"):
                    fontes_html = ""
                    vistos = set()
                    for doc in resultado["documentos"]:
                        src   = doc.metadata.get("source_file", "?")
                        pag   = doc.metadata.get("page", "?")
                        chave = f"{src}_{pag}"
                        if chave not in vistos:
                            fontes_html += f'<span class="fonte-pill">{src} — pág.{pag}</span>'
                            vistos.add(chave)
                    st.markdown(fontes_html, unsafe_allow_html=True)
        else:
            st.warning("Digite uma pergunta para continuar.")

# ══════════════════════════════
# ABA 2 — Plano Preventivo
# ══════════════════════════════
with aba2:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card info" style="margin-bottom:1.5rem;margin-top:0;">
    Preencha seu perfil e o PrevIA consultará as diretrizes do Ministério da Saúde, INCA e OMS
    para gerar um plano preventivo anual com exames, vacinas, hábitos e cronograma personalizado.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="field-label">Idade</div>', unsafe_allow_html=True)
        idade = st.number_input(
            "", min_value=1, max_value=110, value=35,
            label_visibility="collapsed", key="inp_idade"
        )
        st.markdown('<div class="field-label" style="margin-top:1rem;">Sexo</div>', unsafe_allow_html=True)
        sexo = st.selectbox(
            "", ["feminino", "masculino"],
            label_visibility="collapsed", key="inp_sexo"
        )

    with col2:
        st.markdown('<div class="field-label">Histórico familiar</div>', unsafe_allow_html=True)
        historico = st.text_input(
            "", placeholder="Ex: diabetes, hipertensão, câncer de mama",
            label_visibility="collapsed", key="inp_hist"
        )
        st.markdown('<div class="field-label" style="margin-top:1rem;">Condições atuais</div>', unsafe_allow_html=True)
        condicoes = st.text_input(
            "", placeholder="Ex: sedentário, fumante, gestante",
            label_visibility="collapsed", key="inp_cond"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn2, _ = st.columns([1, 4])
    with col_btn2:
        gerar = st.button("Gerar Plano →", type="primary", key="btn_plano", use_container_width=True)

    if gerar:
        perfil = {
            "idade":    idade,
            "sexo":     sexo,
            "historico": historico or "nenhum",
            "condicoes": condicoes or "nenhuma",
        }
        with st.spinner("Gerando seu plano preventivo..."):
            estado = {
                "pergunta": "Gere meu plano preventivo anual personalizado",
                "perfil": perfil, "rota": "automacao", "documentos": [],
                "resposta": "", "check_ok": False, "tentativas": 0, "resposta_final": ""
            }
            resultado = app_graph.invoke(estado)

        partes    = resultado["resposta_final"].split("---")
        plano_txt = partes[0].strip()
        aviso_txt = partes[1].strip() if len(partes) > 1 else ""

        st.markdown(f'<div class="card">{plano_txt}</div>', unsafe_allow_html=True)
        if aviso_txt:
            st.markdown(f'<div class="aviso">{aviso_txt}</div>', unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            with st.expander("📄 Documentos consultados"):
                vistos, fontes_html = set(), ""
                for doc in resultado["documentos"]:
                    src   = doc.metadata.get("source_file", "?")
                    pag   = doc.metadata.get("page", "?")
                    chave = f"{src}_{pag}"
                    if chave not in vistos:
                        fontes_html += f'<span class="fonte-pill">{src} — pág.{pag}</span>'
                        vistos.add(chave)
                st.markdown(fontes_html, unsafe_allow_html=True)
        with col_e2:
            with st.expander("📋 Log de auditoria MCP"):
                try:
                    with open("/content/previa/mcp_audit.log") as f:
                        conteudo = f.read()
                    st.code(
                        conteudo[-2000:] if len(conteudo) > 2000 else conteudo,
                        language="text"
                    )
                except Exception:
                    st.write("Log vazio.")