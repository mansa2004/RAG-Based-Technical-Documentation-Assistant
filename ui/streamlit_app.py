"""
Minimal Streamlit UI for the RAG assistant.

This is a thin client over the FastAPI backend -- it does not import any src/ module or
touch the graph/vector store directly, only HTTP calls to the running API. That keeps
the UI fully decoupled: it works identically against a local uvicorn instance or a
deployed one, and the backend has no dependency on Streamlit at all.

Run:
    uvicorn src.api.main:app --reload          # in one terminal
    streamlit run ui/streamlit_app.py           # in another

Configure a different backend with:
    API_BASE_URL=http://your-host:8000 streamlit run ui/streamlit_app.py
"""
import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar: corpus status + ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Indexed documents")
    try:
        docs = requests.get(f"{API_BASE_URL}/documents", timeout=10).json()
        total_sources = docs.get("total_sources", 0)
        total_chunks = docs.get("total_chunks", 0)
        st.caption(f"{total_sources} source(s), {total_chunks} chunk(s) indexed")
        for d in docs.get("documents", []):
            st.write(f"- **{d['source']}** ({d['chunk_count']} chunks)")
        api_reachable = True
    except Exception as e:
        st.error(f"Could not reach API at {API_BASE_URL}\n\n{e}")
        api_reachable = False

    st.divider()
    st.subheader("Add a document")

    uploaded = st.file_uploader("Upload .md / .txt / .html", type=["md", "txt", "html"])
    if uploaded is not None:
        with st.spinner("Ingesting..."):
            try:
                files = {
                    "files": (
                        uploaded.name,
                        uploaded.getvalue(),
                    )
                }

                r = requests.post(
                    f"{API_BASE_URL}/ingest",
                    files=files,
                    timeout=60,
                )

                if r.ok:
                    st.success(r.json().get("message", "Ingested."))
                else:
                    st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")

            except Exception as e:
                st.error(str(e))

    url_input = st.text_input("...or ingest from a URL")
    if st.button("Ingest URL", disabled=not url_input):
        with st.spinner("Fetching and ingesting..."):
            try:
                r = requests.post(f"{API_BASE_URL}/ingest", data={"urls": url_input}, timeout=60)
                if r.ok:
                    st.success(r.json().get("message", "Ingested."))
                else:
                    st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")
            except Exception as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# Main area: ask a question
# ---------------------------------------------------------------------------
st.title("📚 Technical Documentation RAG Assistant")
st.caption("Self-corrective RAG built with LangGraph + FastAPI. Ask a question about the indexed docs.")

if "last_response" not in st.session_state:
    st.session_state.last_response = None

question = st.text_input("Your question", placeholder="How do I define a request body in FastAPI?")
ask_clicked = st.button("Ask", type="primary", disabled=not question)

if ask_clicked and question:
    with st.spinner("Running the graph (analyze → retrieve → grade → generate)..."):
        try:
            r = requests.post(f"{API_BASE_URL}/query", json={"question": question}, timeout=120)
            if r.ok:
                st.session_state.last_response = r.json()
            else:
                st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")
                st.session_state.last_response = None
        except Exception as e:
            st.error(str(e))
            st.session_state.last_response = None

resp = st.session_state.last_response
if resp:
    st.markdown("### Answer")
    st.write(resp["answer"])

    if resp.get("sources"):
        st.markdown("**Sources:** " + ", ".join(resp["sources"]))

    cols = st.columns(4)
    cols[0].metric("Query type", resp.get("query_type") or "—")
    cols[1].metric("Retries used", resp.get("retries_used", 0))
    grounded = resp.get("answer_is_grounded")
    cols[2].metric("Grounded", "—" if grounded is None else ("Yes" if grounded else "No"))
    cols[3].metric("Web fallback used", "Yes" if resp.get("used_web_fallback") else "No")

    with st.expander("Graded documents"):
        gd = resp.get("graded_documents", [])
        if gd:
            st.table(
                [
                    {
                        "source": g["source"],
                        "relevant": g["relevant"],
                        "reasoning": g["reasoning"],
                    }
                    for g in gd
                ]
            )
        else:
            st.caption("No graded documents in this response.")

    with st.expander("Execution trace"):
        for line in resp.get("trace", []):
            st.text(line)

    st.divider()
    st.markdown("**Was this helpful?**")
    fb_cols = st.columns([1, 1, 6])
    if fb_cols[0].button("👍"):
        requests.post(
            f"{API_BASE_URL}/feedback",
            json={"question": resp["question"], "answer": resp["answer"], "rating": "up"},
            timeout=10,
        )
        st.toast("Thanks for the feedback!")
    if fb_cols[1].button("👎"):
        requests.post(
            f"{API_BASE_URL}/feedback",
            json={"question": resp["question"], "answer": resp["answer"], "rating": "down"},
            timeout=10,
        )
        st.toast("Thanks for the feedback!")
