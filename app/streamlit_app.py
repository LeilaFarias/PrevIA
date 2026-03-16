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
DISCLAIMER = """\n---\n> ⚠️ **Aviso:** Informações educativas baseadas em diretrizes públicas do Ministério da Saúde e OMS. Não substitui consulta com profissional de saúde."""

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
# UI — Design clínico-editorial
# ════════════════════════════════════════════════════════════

st.set_page_config(page_title="PrevIA", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }

.stApp { background: #f5f2ed; }

.hero {
    background: #0f2318;
    margin: -4rem -4rem 2.5rem;
    padding: 3rem 4rem 2.5rem;
    border-bottom: 3px solid #2d7a50;
}
.hero-titulo {
    font-family: "Playfair Display", serif;
    font-size: 3rem;
    color: #e8f0eb;
    margin: 0;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-titulo span { color: #4caf82; }
.hero-sub {
    color: #7aaa8e;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.hero-badges { margin-top: 1.2rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    background: rgba(77,175,130,0.15);
    border: 1px solid rgba(77,175,130,0.3);
    color: #7aaa8e;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1px;
}

.stTabs [data-baseweb="tab-list"] {
    background: white;
    border: 1px solid #dde8e0;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-family: "DM Sans", sans-serif !important;
    font-weight: 500 !important;
    color: #6b8070 !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: #0f2318 !important;
    color: white !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: white !important;
    border: 1.5px solid #dde8e0 !important;
    border-radius: 8px !important;
    font-family: "DM Sans", sans-serif !important;
    font-size: 0.95rem !important;
    color: #1c2520 !important;
    padding: 0.6rem 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2d7a50 !important;
    box-shadow: 0 0 0 3px rgba(45,122,80,0.12) !important;
}

.stButton > button[kind="primary"] {
    background: #0f2318 !important;
    color: #e8f0eb !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: "DM Sans", sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.8rem !important;
    transition: background 0.2s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2d7a50 !important;
}

.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: white !important;
    border: 1.5px solid #dde8e0 !important;
    border-radius: 8px !important;
    font-family: "DM Sans", sans-serif !important;
}

.label-field {
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6b8070;
    margin-bottom: 4px;
}

.card-resposta {
    background: white;
    border: 1px solid #dde8e0;
    border-left: 4px solid #2d7a50;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
    font-size: 0.95rem;
    color: #1c2520;
    line-height: 1.8;
}

.card-plano {
    background: white;
    border: 1px solid #dde8e0;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-top: 1.2rem;
    font-size: 0.95rem;
    color: #1c2520;
    line-height: 1.8;
}

.aviso {
    background: #eef7f2;
    border: 1px solid #b8ddc8;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.82rem;
    color: #4a7060;
    margin-top: 1.2rem;
    line-height: 1.5;
}

.fonte-pill {
    display: inline-block;
    background: #eef7f2;
    border: 1px solid #b8ddc8;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #2d7a50;
    font-weight: 500;
    margin: 3px 3px 0 0;
}

.perfil-card {
    background: white;
    border: 1px solid #dde8e0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

.step-badge {
    background: #0f2318;
    color: #4caf82;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 6px;
}

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #dde8e0 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-sub">Assistente de medicina preventiva</div>
  <div class="hero-titulo">Prev<span>IA</span></div>
  <div class="hero-badges">
    <span class="badge">39 documentos indexados</span>
    <span class="badge">Ministério da Saúde</span>
    <span class="badge">OMS</span>
    <span class="badge">INCA</span>
    <span class="badge">LangGraph + RAG</span>
    <span class="badge">MCP health-checklist</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────
aba1, aba2 = st.tabs(["💬  Perguntas  ", "📋  Plano Preventivo  "])

# ══════════════════════════════
# ABA 1 — Q&A
# ══════════════════════════════
with aba1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_input, col_info = st.columns([3, 1])

    with col_input:
        st.markdown('<div class="label-field">Sua pergunta</div>', unsafe_allow_html=True)
        pergunta = st.text_input("", placeholder="Ex: A partir de qual idade devo fazer mamografia?",
            label_visibility="collapsed", key="input_qa")
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            perguntar = st.button("Perguntar →", type="primary", key="btn_qa", use_container_width=True)

    with col_info:
        st.markdown("""
        <div style="background:white;border:1px solid #dde8e0;border-radius:10px;padding:1rem 1.2rem;font-size:0.8rem;color:#6b8070;line-height:1.7;">
        <strong style="color:#0f2318;font-size:0.82rem;">Como funciona</strong><br><br>
        <span style="color:#2d7a50">①</span> Supervisor classifica<br>
        <span style="color:#2d7a50">②</span> Retriever busca docs<br>
        <span style="color:#2d7a50">③</span> Writer gera resposta<br>
        <span style="color:#2d7a50">④</span> Self-check valida<br>
        <span style="color:#2d7a50">⑤</span> Safety adiciona aviso
        </div>
        """, unsafe_allow_html=True)

    if perguntar:
        if pergunta.strip():
            with st.status("🔍 Processando sua pergunta...", expanded=True) as status:
                st.write("🧭 Analisando intenção...")
                estado = {"pergunta": pergunta, "perfil": None, "rota": "",
                    "documentos": [], "resposta": "", "check_ok": False,
                    "tentativas": 0, "resposta_final": ""}
                estado = supervisor(estado)
                st.write(f"✅ Rota: **{estado['rota'].upper()}**")
                st.write("📚 Buscando nos documentos...")
                estado = retriever_agent(estado)
                fontes_set = list({d.metadata.get("source_file","?") for d in estado["documentos"]})
                st.write(f"✅ {len(estado['documentos'])} trechos em {len(fontes_set)} documentos")
                st.write("✍️ Gerando resposta com citações...")
                estado = writer_agent(estado)
                st.write("🔍 Verificando anti-alucinação...")
                estado = self_check(estado)
                if not estado["check_ok"] and estado["tentativas"] < 2:
                    st.write("⚠️ Re-buscando evidências...")
                    estado = retriever_agent(estado)
                    estado = writer_agent(estado)
                    estado = self_check(estado)
                if estado["check_ok"]:
                    estado = safety_agent(estado)
                else:
                    estado = recusar(estado)
                status.update(label="✅ Resposta pronta!", state="complete", expanded=False)

            partes = estado["resposta_final"].split("---")
            resposta_txt = partes[0].strip()
            aviso_txt = partes[1].strip() if len(partes) > 1 else ""

            st.markdown(f'<div class="card-resposta">{resposta_txt}</div>', unsafe_allow_html=True)

            if aviso_txt:
                st.markdown(f'<div class="aviso">{aviso_txt}</div>', unsafe_allow_html=True)

            if estado.get("documentos"):
                with st.expander("📄 Ver fontes consultadas"):
                    fontes_html = ""
                    vistos = set()
                    for doc in estado["documentos"]:
                        src = doc.metadata.get("source_file","?")
                        pag = doc.metadata.get("page","?")
                        chave = f"{src} pág.{pag}"
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
    <div style="background:white;border:1px solid #dde8e0;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;font-size:0.85rem;color:#4a7060;line-height:1.6;">
    Preencha seu perfil e o PrevIA consultará as diretrizes do Ministério da Saúde, INCA e OMS
    para gerar um plano preventivo anual personalizado — com exames, vacinas, hábitos e cronograma.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="label-field">Idade</div>', unsafe_allow_html=True)
        idade = st.number_input("", min_value=1, max_value=110, value=35,
            label_visibility="collapsed", key="inp_idade")
        st.markdown('<div class="label-field" style="margin-top:1rem;">Sexo</div>', unsafe_allow_html=True)
        sexo = st.selectbox("", ["feminino", "masculino"],
            label_visibility="collapsed", key="inp_sexo")
    with col2:
        st.markdown('<div class="label-field">Histórico familiar</div>', unsafe_allow_html=True)
        historico = st.text_input("", placeholder="Ex: diabetes, hipertensão, câncer de mama",
            label_visibility="collapsed", key="inp_hist")
        st.markdown('<div class="label-field" style="margin-top:1rem;">Condições atuais</div>', unsafe_allow_html=True)
        condicoes = st.text_input("", placeholder="Ex: sedentário, fumante, gestante",
            label_visibility="collapsed", key="inp_cond")

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn2, _ = st.columns([1, 3])
    with col_btn2:
        gerar = st.button("🤖  Gerar Plano Preventivo", type="primary", key="btn_plano", use_container_width=True)

    if gerar:
        perfil = {
            "idade": idade, "sexo": sexo,
            "historico": historico or "nenhum",
            "condicoes": condicoes or "nenhuma"
        }
        with st.status("⏳ Gerando seu plano preventivo...", expanded=True) as status:
            st.write(f"👤 Perfil: {idade} anos, {sexo}, histórico: {historico or 'nenhum'}")
            st.write("🔧 Chamando MCP health-checklist...")
            estado = {
                "pergunta": "Gere meu plano preventivo anual personalizado",
                "perfil": perfil, "rota": "automacao", "documentos": [],
                "resposta": "", "check_ok": False, "tentativas": 0, "resposta_final": ""
            }
            estado = automation_agent(estado)
            st.write(f"✅ {len(estado['documentos'])} documentos consultados")
            st.write("🛡️ Adicionando disclaimer médico...")
            estado = safety_agent(estado)
            status.update(label="✅ Plano gerado!", state="complete", expanded=False)

        partes = estado["resposta_final"].split("---")
        plano_txt = partes[0].strip()
        aviso_txt = partes[1].strip() if len(partes) > 1 else ""

        st.markdown(f'<div class="card-plano">{plano_txt}</div>', unsafe_allow_html=True)
        if aviso_txt:
            st.markdown(f'<div class="aviso">{aviso_txt}</div>', unsafe_allow_html=True)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            with st.expander("📄 Documentos consultados"):
                vistos = set()
                fontes_html = ""
                for doc in estado["documentos"]:
                    src = doc.metadata.get("source_file","?")
                    pag = doc.metadata.get("page","?")
                    chave = f"{src}_{pag}"
                    if chave not in vistos:
                        fontes_html += f'<span class="fonte-pill">{src} — pág.{pag}</span>'
                        vistos.add(chave)
                st.markdown(fontes_html, unsafe_allow_html=True)
        with col_exp2:
            with st.expander("📋 Log de auditoria MCP"):
                try:
                    with open("/content/previa/mcp_audit.log") as f:
                        conteudo = f.read()
                    st.code(conteudo[-2000:] if len(conteudo) > 2000 else conteudo, language="text")
                except:
                    st.write("Log vazio.")
