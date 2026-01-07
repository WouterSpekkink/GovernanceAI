
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import LongContextReorder

# ----------------------------
# Config (env-overridable)
# ----------------------------
PERSIST_DIR = Path(os.getenv("RAG_PERSIST_DIR", "./faiss_index"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")

ANSWER_MODEL = os.getenv("RAG_ANSWER_MODEL", "gpt-5.1")
UTILITY_MODEL = os.getenv("RAG_UTILITY_MODEL", "gpt-5.1")

K_FINAL = int(os.getenv("RAG_K_FINAL", "12"))
K_RETRIEVER = int(os.getenv("RAG_K_RETRIEVER", "24"))

# Cap how much of each doc goes into reranking prompt (keeps it fast/cheap)
RERANK_PASSAGE_CHARS = int(os.getenv("RAG_RERANK_PASSAGE_CHARS", "1200"))


# ----------------------------
# Guardrails: API key must exist (app.py should set it)
# ----------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "In Streamlit, set it via st.secrets['OPENAI_API_KEY'] (or local secrets.toml). "
        "For local runs, you can also set it in a .env file and load it in app.py."
    )


# ----------------------------
# Clients / store
# ----------------------------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# Load FAISS index created by indexer.py
db = FAISS.load_local(
    str(PERSIST_DIR),
    embeddings,
    allow_dangerous_deserialization=True,  # OK if you trust your own index files
)

base_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": K_RETRIEVER},
)

llm_util = ChatOpenAI(model=UTILITY_MODEL)
llm_answer = ChatOpenAI(model=ANSWER_MODEL)

# ----------------------------
# Reranking (optional)
# ----------------------------
RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Score the relevance of the passage to the question on a 0–10 scale. "
            "Return only a number.",
        ),
        ("human", "Question:\n{q}\n\nPassage:\n{p}"),
    ]
)


def _score_passage(q: str, passage: str) -> float:
    msgs = RERANK_PROMPT.format_messages(q=q, p=passage[:RERANK_PASSAGE_CHARS])
    try:
        resp = llm_util.invoke(msgs)
        txt = (resp.content or "").strip()
        # Extract first float-like number
        num = ""
        for ch in txt:
            if ch in "0123456789.":
                num += ch
            elif num:
                break
        return float(num) if num else 0.0
    except Exception:
        return 0.0


def llm_rerank(question: str, docs: List[Document], top_n: int = 12) -> List[Document]:
    scored = [(_score_passage(question, d.page_content), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_n]]


# ----------------------------
# Formatting and sources
# ----------------------------
def get_sources(docs: List[Document]) -> List[dict]:
    out = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        out.append(
            {
                "filename": os.path.basename(src),
                "content": doc.page_content.replace("\n", " "),
            }
        )
    return out


def format_docs(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        fname = os.path.basename(d.metadata.get("source", "unknown"))
        blocks.append(
            f"=== SOURCE {i} ===\n"
            f"File: {fname}\n"
            f"Passage:\n{d.page_content}\n"
        )
    return "\n".join(blocks)


# ----------------------------
# Prompt & LCEL chain
# ----------------------------
SYSTEM_TEMPLATE = """
You are a professor working on governance theories.

You have access to papers a variety of approaches to governance theories.

Based on this context, you will answer various questions that you will get on these different approaches. These questions are from students that are just getting acquainted with governance literature and are trying to identify a governance theory that they can usefully apply to a case that they are attempting to analyze. Your job is to help them navigate this literature. Please be elaborate in your answers.

Use ONLY the provided context to answer. If the context is clearly unrelated or missing the key facts, say you don't know.
If the context is partially relevant, answer what you can from it and explicitly note any gaps.

What is also important: Don't do a full analysis for the students, as they have to do the analysis themselves. However, you can give them good advice on how to approach the analysis of a certain case. Thus, if students ask for a full analysis, respectfully decline, but then give them good advice to help them along with their own analysis.

Chat history (may be empty):
{chat_history}

Context:
{context}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]
)

rag_chain = prompt | llm_answer | StrOutputParser()


# ----------------------------
# Public helpers used by app.py
# ----------------------------
def retrieve_docs(question: str) -> List[Document]:
    """
    Retrieval pipeline:

    1) Similarity retrieval from FAISS
    2) Redundancy filter + reorder for coherence
    3) Truncate to final K
    """
    docs = base_retriever.invoke(question)

    # Optional reranking (expensive). Uncomment if you want it:
    # docs = llm_rerank(question, docs, top_n=max(K_FINAL * 2, len(docs)))

    # Compress/reorder: removes redundant chunks & reorders for readability
    reorderer = LongContextReorder()
    docs = reorderer.transform_documents(docs)

    return docs[:K_FINAL]


def build_context(question: str) -> Tuple[str, List[Document]]:
    docs = retrieve_docs(question)
    context_str = format_docs(docs)
    return context_str, docs
