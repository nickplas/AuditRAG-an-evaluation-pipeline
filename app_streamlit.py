"""
app_streamlit.py — Streamlit web interface for the Auditable RAG Pipeline.
 
Run with:
    streamlit run app_streamlit.py
 
Requires the pipeline to be importable (run from project root).
Uses in_memory=True so no Qdrant server is needed.
"""
 
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st


def render_html(html: str) -> None:
    """
    Render raw HTML via st.markdown without it ever being mistaken for an
    indented code block.

    st.markdown still runs content through a Markdown parser even with
    unsafe_allow_html=True, and 4+ leading spaces on a line make Markdown
    treat it as a literal code block instead of HTML. Multi-line f-strings
    written at deep Python indentation (nested loops/ifs) or with manually
    aligned attribute continuation lines are exactly that case. Flattening
    every line removes the leading whitespace entirely, so the indentation
    of the surrounding Python code can never leak into the rendered page.
    Safe for HTML/CSS: neither cares about whitespace between tags/lines.
    """
    flat = " ".join(line.strip() for line in html.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
 
/* Dark industrial theme */
.stApp {
    background-color: #0f1117;
    color: #e8e8e8;
}
 
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
 
/* Cards */
.rag-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
 
.rag-card-accent {
    border-left: 3px solid #58a6ff;
}
 
.rag-card-warning {
    border-left: 3px solid #f0883e;
}
 
.rag-card-danger {
    border-left: 3px solid #f85149;
}
 
.rag-card-success {
    border-left: 3px solid #3fb950;
}
 
/* Answer text */
.answer-text {
    font-size: 1.05rem;
    line-height: 1.7;
    color: #e8e8e8;
}
 
/* Citation badge */
.citation-badge {
    display: inline-block;
    background: #1f3a5f;
    color: #58a6ff;
    border: 1px solid #1f6feb;
    border-radius: 4px;
    padding: 0px 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}
 
/* Score bar */
.score-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}
 
.score-bar-fill {
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    border-radius: 4px;
    height: 6px;
}
 
/* Monospace for chunk text */
.chunk-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    line-height: 1.6;
    background: #0d1117;
    padding: 0.7rem 1rem;
    border-radius: 6px;
    border: 1px solid #21262d;
    margin-top: 0.5rem;
}
 
/* Metric boxes */
.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
 
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
}
 
.metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
 
/* Refused banner */
.refused-banner {
    background: #1f1410;
    border: 1px solid #f85149;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    color: #f85149;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
}
 
/* KG flag */
.kg-flag {
    background: #1f1a0e;
    border: 1px solid #f0883e;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    color: #f0883e;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
 
/* Header */
.main-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.02em;
}
 
.sub-header {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: -0.5rem;
    margin-bottom: 1.5rem;
}
 
/* Input box */
.stTextInput > div > div > input,
.stTextArea textarea {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 6px !important;
}
 
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}
 
/* Buttons */
.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s !important;
}
 
.stButton > button:hover {
    background: #388bfd !important;
}
 
/* File uploader */
.stFileUploader {
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 0.5rem;
}
 
/* Divider */
hr {
    border-color: #21262d !important;
    margin: 1.5rem 0 !important;
}
 
/* Expander */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
 
/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
 
# ── Pipeline initialisation (cached) ──────────────────────────────────────────
 
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.pipeline import RAGPipeline
    return RAGPipeline(in_memory=True, use_knowledge_graph=False)
 
 
# ── Session state ─────────────────────────────────────────────────────────────
 
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "history" not in st.session_state:
    st.session_state.history = []   # list of {query, response}
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_refused" not in st.session_state:
    st.session_state.total_refused = 0
if "latencies" not in st.session_state:
    st.session_state.latencies = []
if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = []
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
 
with st.sidebar:
    st.markdown('<div class="main-header">⬡ RAG Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Auditable · Grounded · Trustworthy</div>', unsafe_allow_html=True)
    st.markdown("---")
 
    # ── Document indexing ──────────────────────────────────────────────────
    st.markdown("#### 📂 Index Documents")
    st.caption("Supported: PDF · Word · Excel · PowerPoint · HTML · CSV · Markdown · TXT")
 
    # Option A: upload files
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "md", "pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "html", "htm", "csv", "json", "xml", "epub"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
 
    # Option B: index from data/ directory
    use_data_dir = st.checkbox("Index from `data/` folder", value=False)
 
    if st.button("Index documents", use_container_width=True):
        if not uploaded_files and not use_data_dir:
            st.warning("Upload files or check 'Index from data/ folder'.")
        else:
            with st.spinner("Loading pipeline…"):
                pipeline = load_pipeline()
                st.session_state.pipeline = pipeline
 
            with st.spinner("Indexing…"):
                sources = []
 
                if uploaded_files:
                    import tempfile
                    tmp_dir = Path(tempfile.mkdtemp())
                    for uf in uploaded_files:
                        dest = tmp_dir / uf.name
                        dest.write_bytes(uf.read())
                        sources.append(uf.name)
                    pipeline.index(tmp_dir)
 
                if use_data_dir:
                    data_path = Path("data")
                    if data_path.exists():
                        pipeline.index(data_path)
                        sources += [
                            p.name for p in data_path.rglob("*")
                            if p.suffix in {".txt", ".pdf", ".json"}
                        ]
                    else:
                        st.error("`data/` folder not found.")
 
                st.session_state.indexed = True
                st.session_state.indexed_sources = sources
 
            st.success(f"Indexed {pipeline.collection_size} chunks.")
 
    # Show indexed sources
    if st.session_state.indexed_sources:
        st.markdown("**Indexed sources:**")
        for src in st.session_state.indexed_sources:
            st.markdown(f"- `{src}`")
 
    st.markdown("---")
 
    # ── Retrieval settings ─────────────────────────────────────────────────
    st.markdown("#### ⚙️ Retrieval Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 5)
    score_threshold = st.slider("Min relevance score", 0.0, 1.0, 0.35, 0.05)
 
    st.markdown("---")
 
    # ── Session stats ──────────────────────────────────────────────────────
    st.markdown("#### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Refused", st.session_state.total_refused)
 
    if st.session_state.latencies:
        import statistics
        st.metric(
            "Median latency",
            f"{statistics.median(st.session_state.latencies):.0f} ms"
        )
 
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_queries = 0
        st.session_state.total_refused = 0
        st.session_state.latencies = []
        st.rerun()
 
 
# ── Main area ─────────────────────────────────────────────────────────────────
 
if not st.session_state.indexed:
    # Welcome screen
    render_html("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:3rem; color:#58a6ff;">⬡</div>
        <h2 style="font-family:'IBM Plex Mono',monospace; color:#e8e8e8; margin-top:1rem;">
            Auditable RAG Pipeline
        </h2>
        <p style="color:#8b949e; max-width:480px; margin:1rem auto; line-height:1.7;">
            Upload your documents in the sidebar to get started.
            Every answer comes with citations, a confidence score,
            and full source provenance.
        </p>
        <div style="display:flex; justify-content:center; gap:2rem; margin-top:2rem;">
            <div class="rag-card" style="width:180px;">
                <div style="font-size:1.5rem">📎</div>
                <div style="font-size:0.8rem;color:#8b949e;margin-top:0.5rem;">Upload .txt or .pdf files</div>
            </div>
            <div class="rag-card" style="width:180px;">
                <div style="font-size:1.5rem">🔍</div>
                <div style="font-size:0.8rem;color:#8b949e;margin-top:0.5rem;">Ask questions in natural language</div>
            </div>
            <div class="rag-card" style="width:180px;">
                <div style="font-size:1.5rem">🔗</div>
                <div style="font-size:0.8rem;color:#8b949e;margin-top:0.5rem;">Audit every answer with citations</div>
            </div>
        </div>
    </div>
    """)
 
else:
    # ── Query input ────────────────────────────────────────────────────────
    st.markdown("### Ask a question")
 
    with st.form("query_form", clear_on_submit=True):
        query_input = st.text_input(
            "Question",
            placeholder="e.g. What are the main risks described in the document?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", use_container_width=False)
 
    if submitted and query_input.strip():
        pipeline = st.session_state.pipeline
 
        # Temporarily override threshold from slider
        from src.config import settings as cfg
        cfg.score_threshold = score_threshold
 
        with st.spinner("Retrieving and generating…"):
            response = pipeline.query(query_input.strip(), top_k=top_k)
 
        # Update session stats
        st.session_state.total_queries += 1
        if response.refused:
            st.session_state.total_refused += 1
        st.session_state.latencies.append(response.latency_ms)
 
        # Prepend to history (newest first)
        st.session_state.history.insert(0, {
            "query": query_input.strip(),
            "response": response,
        })
 
    # ── History ────────────────────────────────────────────────────────────
    for i, item in enumerate(st.session_state.history):
        query = item["query"]
        resp = item["response"]
 
        render_html(f"""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
                    color:#8b949e; margin-bottom:0.3rem;">
            Q #{st.session_state.total_queries - i}
        </div>
        <div style="font-size:1.05rem; font-weight:600; color:#e8e8e8;
                    margin-bottom:0.8rem;">
            {query}
        </div>
        """)
 
        if resp.refused:
            render_html(f"""
            <div class="refused-banner">
                ⚠ REFUSED — {resp.refusal_reason}
            </div>
            """)
 
        else:
            # ── Answer card ────────────────────────────────────────────────
            conf_color = "#3fb950" if resp.confidence >= 0.6 else \
                         "#f0883e" if resp.confidence >= 0.35 else "#f85149"
            conf_pct = int(resp.confidence * 100)
 
            render_html(f"""
            <div class="rag-card rag-card-accent">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:0.8rem;">
                    <span style="font-family:'IBM Plex Mono',monospace;
                                 font-size:0.72rem; color:#8b949e;
                                 text-transform:uppercase; letter-spacing:0.08em;">
                        Answer
                    </span>
                    <span style="font-family:'IBM Plex Mono',monospace;
                                 font-size:0.8rem; color:{conf_color};">
                        confidence {conf_pct}%
                    </span>
                </div>
                <div class="answer-text">{resp.answer}</div>
            </div>
            """)
 
            # ── KG flags ───────────────────────────────────────────────────
            if resp.kg_flags:
                render_html(f"""
                <div class="kg-flag">
                    ⚠ KG flag — entities not grounded in corpus:
                    <strong>{', '.join(resp.kg_flags)}</strong>
                </div>
                """)
 
            # ── Citations ──────────────────────────────────────────────────
            if resp.citations:
                with st.expander(
                    f"📎  {len(resp.citations)} source chunk(s) — click to audit",
                    expanded=(i == 0),
                ):
                    for c in resp.citations:
                        score_pct = int(c['score'] * 100)
                        score_color = "#3fb950" if c['score'] >= 0.6 else \
                                      "#f0883e" if c['score'] >= 0.35 else "#f85149"
 
                        page_str = f" · page {c['page']}" if c.get('page') else ""
                        source_short = Path(c['source']).name if c['source'] else c['source']
                        section_str = c.get('section') or c.get('metadata', {}).get('section', '')
                        section_html = (
                            f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.72rem;'
                            f' color:#58a6ff; margin-top:4px;">§ {section_str}</div>'
                            if section_str else ''
                        )
 
                        render_html(f"""
                        <div class="rag-card" style="margin-bottom:0.7rem;">
                            <div style="display:flex; justify-content:space-between;
                                        align-items:baseline;">
                                <span>
                                    <span class="citation-badge">[{c['index']}]</span>
                                    <span style="font-family:'IBM Plex Mono',monospace;
                                                 font-size:0.78rem; color:#8b949e;">
                                        {source_short}{page_str}
                                    </span>
                                </span>
                                <span style="font-family:'IBM Plex Mono',monospace;
                                             font-size:0.78rem; color:{score_color};">
                                    {score_pct}% match
                                </span>
                            </div>
                            {section_html}
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width:{score_pct}%;
                                     background: linear-gradient(90deg, #1f6feb, {score_color});"></div>
                            </div>
                            <div class="chunk-text">{c['text']}</div>
                        </div>
                        """)
 
            # ── Sources ────────────────────────────────────────────────────
            if resp.sources:
                sources_str = " · ".join(Path(s).name for s in resp.sources)
                render_html(f"""
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
                            color:#8b949e; margin-top:0.3rem;">
                    Sources: {sources_str} · {resp.latency_ms:.0f} ms
                </div>
                """)
 
        if i < len(st.session_state.history) - 1:
            st.markdown("---")