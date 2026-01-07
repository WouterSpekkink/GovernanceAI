from __future__ import annotations

import hashlib
import os
from typing import List, Tuple, Iterable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

# Dedupe/compress utilities
from langchain_community.document_transformers import LongContextReorder, EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# ----------------------------
# Config
# ----------------------------
PERSIST_DIR = "./faiss_index"   # where indexer.py saved the FAISS index

# Embeddings
EMBED_MODEL = "text-embedding-3-large"

# LLMs
ANSWER_MODEL = "gpt-5.1"        # main answering model
UTILITY_MODEL = "gpt-5.1"

# Retrieval knobs
K_FINAL = 12          # number of chunks to keep after rerank/compress
K_RETRIEVER = 24      # how many chunks to retrieve before truncation / compression

# ----------------------------
# Env / clients
# ----------------------------
load_dotenv()
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# Load FAISS index created by indexer.py
db = FAISS.load_local(
    PERSIST_DIR,
    embeddings,
    allow_dangerous_deserialization=True,  # required because index.pkl uses pickle
)

# Base retriever (similarity search; good for small corpus)
base_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": K_RETRIEVER,
    },
)

# Utility & answer models
llm_util = ChatOpenAI(model=UTILITY_MODEL)
llm_answer = ChatOpenAI(model=ANSWER_MODEL)

# Light compression (after rerank) – you’re currently only using reorderer in retrieve_docs
redundancy_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
reorderer = LongContextReorder()
compressor = DocumentCompressorPipeline(transformers=[redundancy_filter, reorderer])

# ----------------------------
# LLM re-ranking (compact numeric)
# ----------------------------
RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Score the relevance of the passage to the question on a 0–10 scale. "
     "Return only a number."),
    ("human", "Question:\n{q}\n\nPassage:\n{p}")
])

def _score_passage(q: str, passage: str) -> float:
    msgs = RERANK_PROMPT.format_messages(q=q, p=passage[:1200])  # cap per doc to keep it fast
    try:
        resp = llm_util.invoke(msgs)
        txt = resp.content.strip()
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
    scored: List[Tuple[float, Document]] = []
    for d in docs:
        s = _score_passage(question, d.page_content)
        scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_n]]

# ----------------------------
# Formatting and sources
# ----------------------------
def get_sources(docs: List[Document]) -> List[dict]:
    """Return structured source info for UI."""
    out = []
    for doc in docs:
        m = doc.metadata
        out.append({
            "filename": os.path.basename(m.get("source", "unknown")),
            "content": doc.page_content.replace("\n", " "),
        })
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

def _cleanup(s: str) -> str:
    return (s or "").replace("{", "").replace("}", "").replace("\\", "").replace("/", "")

# ----------------------------
# Prompt & LCEL answer chain
# ----------------------------
SYSTEM_TEMPLATE = """
You are a professor working on organizational management.

You have access to papers on the relative benefits of working online and working on-site. Context from these papers will be provided to you.

Based on this context, you will answer various questions that you will get on this topic. Please be elaborate in your answers.

Use ONLY the provided context to answer. If the context is clearly unrelated or missing the key facts, say you don't know.
If the context is partially relevant, answer what you can from it and explicitly note any gaps.

Chat history (may be empty):
{chat_history}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate.from_template(SYSTEM_TEMPLATE)),
    HumanMessagePromptTemplate.from_template("{question}")
])

# LCEL chain that expects {"question","context","chat_history"} and returns a string
rag_chain = (prompt | llm_answer | StrOutputParser())

# ----------------------------
# Public helpers used by app.py
# ----------------------------
def retrieve_docs(question: str) -> List[Document]:
    """
    Retrieval pipeline optimized for a small number of papers.

    1) Similarity search from FAISS
    2) Optional LLM-based re-ranking (currently disabled)
    3) Light reordering for coherence
    4) Truncate to final K
    """
    # 1) Initial similarity retrieval
    docs = base_retriever.invoke(question)

    # 2) Optional: LLM re-ranking – you can enable this if you want
    # docs = llm_rerank(question, docs, top_n=max(K_FINAL * 2, len(docs)))

    # 3) Light reordering (no extra embeddings)
    docs = reorderer.transform_documents(docs)

    # 4) Keep final K
    return docs[:K_FINAL]

def build_context(question: str) -> Tuple[str, List[Document]]:
    """Fetch docs and build the context string."""
    docs = retrieve_docs(question)
    return format_docs(docs), docs
