import os
import signal
from datetime import datetime
import streamlit as st

if "APIKEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["APIKEY"]

from rag_chain import rag_chain, build_context, get_sources  # from your rag_chain.py

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="BernAId", layout="wide")
st.title("BernAId")

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.markdown("### Session")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
        
# ----------------------------
# Render chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"`{s['filename']}`")
                    st.markdown(f"> {s['content'][:1000]}{'…' if len(s['content'])>1000 else ''}")
                    st.markdown("---")

#-----------------------------
# Chatbot memory
#-----------------------------
def build_chat_history():
    history_lines = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            history_lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            history_lines.append(f"Assistant: {m['content']}")
    return "\n".join(history_lines[-8:])  # keep last ~8 turns

# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Ask a question…")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant turn
    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer…"):
            # Build retrieval context and get sources
            context_str, docs = build_context(user_input)
            sources = get_sources(docs)

            # Call the LCEL chain (expects question, context, chat_history)
            chat_history = build_chat_history()

            answer = rag_chain.invoke({
                "question": user_input,
                "context": context_str,
                "chat_history": chat_history
            })
            
            # Compose full message with a Sources section beneath
            full_answer = answer
            if sources:
                full_answer += "\n\n### Sources\n"
                for idx, src in enumerate(sources, 1):
                    full_answer += f"`{src['filename']}`\n\n"
                    # Keep the preview compact in the log
                    preview = src["content"].replace("\n", " ")
                    if len(preview) > 1200:
                        preview = preview[:1200] + "…"
                    full_answer += f"> {preview}\n\n"

            # Render to UI
            st.markdown(answer)
            if sources:
                with st.expander("Sources", expanded=False):
                    for i, s in enumerate(sources, 1):
                        st.markdown(f"`{s['filename']}`")
                        st.markdown(f"> {s['content'][:1500]}{'…' if len(s['content'])>1500 else ''}")
                        st.markdown("---")
            else:
                st.info("No relevant documents were retrieved for this query.")

            # Append to session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

