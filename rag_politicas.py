"""Chat RAG didático: políticas de RH (férias e home-office) com DeepSeek + FAISS + Streamlit."""

import hashlib
import os
import shutil

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
load_dotenv()  # lê DEEPSEEK_API_KEY do arquivo .env

PDF_PADRAO = "docs/politicas.pdf"
PDF_UPLOAD = "docs/politicas_upload.pdf"
INDICE_DIR = "./faiss_index"
ARQ_HASH = os.path.join(INDICE_DIR, "fonte.sha256")
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
MSG_SEM_RESPOSTA = "Não encontrei essa informação nas políticas indexadas."

PROMPT_SISTEMA = f"""Você é um assistente de políticas de RH de uma empresa.
Responda SOMENTE com base no contexto abaixo, em português do Brasil, de forma clara e objetiva.
Se a resposta não estiver no contexto, responda exatamente: "{MSG_SEM_RESPOSTA}"

Contexto:
{{context}}"""

# CSS para o visual estilo DeepSeek: mensagens do usuário à direita.
CSS = """
<style>
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    text-align: right;
}
[data-testid="stChatMessage"] { border-radius: 12px; }
</style>
"""


# ---------------------------------------------------------------------------
# Pipeline RAG
# ---------------------------------------------------------------------------
def hash_pdf(caminho_pdf: str | None) -> str | None:
    """Devolve o SHA-256 do conteúdo do PDF (ou None se o arquivo não existir)."""
    if not caminho_pdf or not os.path.exists(caminho_pdf):
        return None
    with open(caminho_pdf, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def hash_do_indice() -> str | None:
    """Lê o hash do PDF que gerou o índice salvo em disco."""
    try:
        with open(ARQ_HASH) as f:
            return f.read().strip()
    except OSError:
        return None


@st.cache_resource(show_spinner="Carregando índice vetorial...")
def carregar_ou_criar_vectorstore(caminho_pdf: str, assinatura: str | None):
    """Carrega o índice FAISS do disco; recria se o PDF de origem mudou.

    `assinatura` (hash do PDF) faz parte da chave de cache: trocar o PDF
    invalida o cache do Streamlit automaticamente.
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)

    # Índice existe e veio do mesmo PDF (ou não há PDF para comparar): reutiliza.
    if os.path.isdir(INDICE_DIR) and (
        assinatura is None or assinatura == hash_do_indice()
    ):
        return FAISS.load_local(
            INDICE_DIR, embeddings, allow_dangerous_deserialization=True
        )

    # Fonte mudou: descarta o índice antigo e reprocessa.
    shutil.rmtree(INDICE_DIR, ignore_errors=True)

    # 1) Loader: lê as páginas do PDF.
    paginas = PyPDFLoader(caminho_pdf).load()
    # 2) Splitter: quebra em trechos de 800 caracteres com 100 de sobreposição.
    trechos = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    ).split_documents(paginas)
    # 3) Vector store: gera embeddings e persiste em disco.
    vectorstore = FAISS.from_documents(trechos, embeddings)
    vectorstore.save_local(INDICE_DIR)
    with open(ARQ_HASH, "w") as f:
        f.write(assinatura or "")
    return vectorstore


@st.cache_resource(show_spinner=False)
def construir_chain(_vectorstore):
    """Monta a chain de recuperação (retriever + prompt + DeepSeek)."""
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_SISTEMA),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    combinar_docs = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combinar_docs)


def responder(chain, pergunta: str, historico: list[dict]):
    """Envia a pergunta (com o histórico) à chain e devolve (resposta, trechos)."""
    mensagens = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in historico
    ]
    resultado = chain.invoke({"input": pergunta, "chat_history": mensagens})
    return resultado["answer"], resultado["context"]


# ---------------------------------------------------------------------------
# Auxiliares de interface
# ---------------------------------------------------------------------------
def reindexar():
    """Apaga o índice em disco e limpa o cache para reprocessar o PDF."""
    shutil.rmtree(INDICE_DIR, ignore_errors=True)
    st.cache_resource.clear()


def mostrar_fontes(fontes: list[dict]):
    """Exibe os trechos consultados dentro de um expander."""
    with st.expander("📄 Fontes consultadas"):
        for f in fontes:
            st.caption(f"Página {f['pagina']}")
            st.write(f["texto"])


def sidebar() -> str | None:
    """Renderiza a sidebar e devolve o caminho do PDF a ser usado (ou None)."""
    with st.sidebar:
        st.header("Opções")
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("🔄 Reindexar PDF", use_container_width=True):
            reindexar()
            st.rerun()

        upload = st.file_uploader("PDF substituto (opcional)", type="pdf")
        if upload is not None:
            # Salva o upload; a reindexação é automática pela mudança de hash.
            with open(PDF_UPLOAD, "wb") as f:
                f.write(upload.getbuffer())
            return PDF_UPLOAD

    return PDF_PADRAO if os.path.exists(PDF_PADRAO) else None


# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------
def main():
    """Ponto de entrada: configura a página, valida o ambiente e roda o chat."""
    st.set_page_config(page_title="Políticas RH • Chat", page_icon="💬", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Assistente de Políticas de RH")
    st.caption("Pergunte sobre férias e home-office.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    caminho_pdf = sidebar()

    # Validações de ambiente.
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.error("DEEPSEEK_API_KEY não encontrada. Copie `.env.example` para `.env` e informe sua chave.")
        st.stop()
    if caminho_pdf is None and not os.path.isdir(INDICE_DIR):
        st.error(f"PDF não encontrado. Coloque `{PDF_PADRAO}` na raiz do projeto ou envie um pela barra lateral.")
        st.stop()

    try:
        vectorstore = carregar_ou_criar_vectorstore(
            caminho_pdf or PDF_PADRAO, hash_pdf(caminho_pdf)
        )
        chain = construir_chain(vectorstore)
    except Exception as e:
        st.error(f"Erro ao preparar o índice: {e}")
        st.stop()

    # Histórico persistido na sessão.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("fontes"):
                mostrar_fontes(msg["fontes"])

    # Nova pergunta (o campo de input fica fixo no rodapé).
    pergunta = st.chat_input("Digite sua pergunta...")
    if not pergunta:
        return

    historico = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        fontes = []
        with st.spinner("Consultando as políticas..."):
            try:
                resposta, docs = responder(chain, pergunta, historico)
                fontes = [
                    {"pagina": d.metadata.get("page", 0) + 1, "texto": d.page_content}
                    for d in docs
                ]
            except Exception as e:
                resposta = f"⚠️ Ocorreu um erro ao consultar o modelo: {e}"
        st.markdown(resposta)
        if fontes:
            mostrar_fontes(fontes)

    st.session_state.messages.append(
        {"role": "assistant", "content": resposta, "fontes": fontes}
    )


if __name__ == "__main__":
    main()
