"""
app.py — AI Data Analyst (₹0 cost, local Qwen via Ollama)

Flow:
CSV/Excel -> Pandas processing -> AI Analyst (Qwen + Ollama)
    -> Statistics | Charts | Insights | Q&A Chat

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_processor import load_file, basic_stats, correlation_matrix, build_context_summary
from ollama_client import ask_ollama, check_ollama_alive, DEFAULT_MODEL
from query_engine import answer_question

st.set_page_config(page_title="AI Data Analyst", page_icon="◆", layout="wide")

# ---------------------------------------------------------------
# Custom CSS — dashboard / product look, not a demo script
# ---------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 18px; }
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    p, li, span, label, div { font-size: 1rem; }

    /* ---------- Top bar ---------- */
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 18px 6px 22px 6px; border-bottom: 1px solid #E5E7EB; margin-bottom: 26px;
    }
    .topbar-left { display: flex; align-items: center; gap: 16px; }
    .logo-mark {
        width: 48px; height: 48px; border-radius: 11px;
        background: linear-gradient(135deg, #0D9488, #1E40AF);
        color: white; display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1.4rem;
    }
    .topbar-title { font-size: 1.6rem; font-weight: 800; color: #111827; line-height: 1.15; }
    .topbar-sub { font-size: 1rem; color: #6B7280; font-weight: 500; }
    .pill {
        font-size: 0.95rem; font-weight: 600; padding: 7px 16px; border-radius: 20px;
        display: inline-flex; align-items: center; gap: 7px;
    }
    .pill-live { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .pill-off  { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
    .pill-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

    /* ---------- Section labels ---------- */
    .panel-title {
        font-size: 1.15rem; font-weight: 700; color: #111827; text-transform: uppercase;
        letter-spacing: 0.04em; margin-bottom: 6px;
    }
    .panel-sub { font-size: 1rem; color: #6B7280; margin-bottom: 18px; }

    /* ---------- KPI stat cards ---------- */
    .kpi-card {
        border: 1px solid #E5E7EB; border-radius: 14px; padding: 20px 22px;
        background: #FFFFFF; height: 100%;
    }
    .kpi-label { font-size: 0.92rem; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.03em; }
    .kpi-value { font-size: 2.2rem; font-weight: 800; color: #111827; margin-top: 6px; }
    .kpi-accent { height: 4px; width: 32px; border-radius: 3px; background: #0D9488; margin-top: 12px; }

    /* ---------- Native container-as-panel ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important; border: 1px solid #E5E7EB !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E5E7EB; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 14px 24px; font-weight: 600; font-size: 1.08rem;
        background: transparent; color: #6B7280;
    }
    .stTabs [aria-selected="true"] {
        background: #F0FDFA !important; color: #0D9488 !important; border-bottom: 3px solid #0D9488 !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 9px; font-weight: 600; font-size: 1.05rem; border: 1px solid #0D9488;
        background: #0D9488; color: white; padding: 11px 26px; transition: all 0.15s ease;
    }
    .stButton > button:hover { background: #0B7A70; border-color: #0B7A70; }

    /* ---------- File uploader ---------- */
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 10px; border: 1.5px dashed #D1D5DB; background: #FAFAFA;
    }
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] * { font-size: 0.95rem !important; }

    /* ---------- Chat ---------- */
    div[data-testid="stChatMessage"] {
        border-radius: 12px; border: 1px solid #EEF1F6; font-size: 1.05rem;
    }

    /* ---------- Dataframes ---------- */
    div[data-testid="stDataFrame"] * { font-size: 0.98rem !important; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { border-right: 1px solid #E5E7EB; background: #FAFAFA; width: 340px !important; }
    section[data-testid="stSidebar"] h1 { font-size: 1.3rem !important; font-weight: 700 !important; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown { font-size: 1.02rem !important; }
    section[data-testid="stSidebar"] input { font-size: 1rem !important; }

    /* ---------- Footer note ---------- */
    .footnote { text-align: center; color: #9CA3AF; font-size: 0.92rem; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar — data source + model config
# ---------------------------------------------------------------
st.sidebar.title("Configuration")

st.sidebar.markdown("**Data source**")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"],
                                          label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model**")
model_name = st.sidebar.text_input("Engine version", value=DEFAULT_MODEL, label_visibility="collapsed",
                                    help="Identifier of the installed local AI engine")
ollama_ok = check_ollama_alive()
if ollama_ok:
    st.sidebar.markdown('<span class="pill pill-live"><span class="pill-dot"></span>AI engine online</span>',
                         unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="pill pill-off"><span class="pill-dot"></span>AI engine offline</span>',
                         unsafe_allow_html=True)
    st.sidebar.caption("Start the local AI engine from your terminal, then refresh this page.")

st.sidebar.markdown("---")
st.sidebar.caption("Runs entirely on this machine. No API key, no external calls, ₹0 marginal cost.")

# ---------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------
status_html = ('<span class="pill pill-live"><span class="pill-dot"></span>AI engine online</span>' if ollama_ok
               else '<span class="pill pill-off"><span class="pill-dot"></span>AI engine offline</span>')

st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="logo-mark">A</div>
        <div>
            <div class="topbar-title">AI Data Analyst</div>
            <div class="topbar-sub">Private, on-device analytics engine</div>
        </div>
    </div>
    <div>{status_html}</div>
</div>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    try:
        df = load_file(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    stats = basic_stats(df)
    context_summary = build_context_summary(df, stats)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visual Analysis", "AI Insights", "Ask AI"])

    # -------------------------------------------------------
    # TAB 1: Overview / Statistics
    # -------------------------------------------------------
    with tab1:
        total_missing = sum(stats["missing_values"].values())

        k1, k2, k3, k4 = st.columns(4)
        for col, label, value in zip(
            [k1, k2, k3, k4],
            ["Rows", "Columns", "Missing values", "Duplicate rows"],
            [stats["rows"], stats["columns"], total_missing, stats["duplicate_rows"]],
        ):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value:,}</div>
                    <div class="kpi-accent"></div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="panel-title">Dataset preview</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">First 20 rows of the uploaded file</div>', unsafe_allow_html=True)
            st.dataframe(df.head(20), use_container_width=True)

        st.write("")
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Missing values</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-sub">Null count per column</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(stats["missing_values"].items(),
                                           columns=["Column", "Missing"]),
                             use_container_width=True, hide_index=True)
        with col_b:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Numeric summary</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-sub">describe() on numeric columns</div>', unsafe_allow_html=True)
                numeric_df = df.select_dtypes(include="number")
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found.")

    # -------------------------------------------------------
    # TAB 2: Charts
    # -------------------------------------------------------
    with tab2:
        plt.rcParams.update({"font.size": 9})
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Distribution</div>', unsafe_allow_html=True)
                col_choice = st.selectbox("Column", numeric_cols, label_visibility="collapsed")
                chart_col, _ = st.columns([2, 1])
                with chart_col:
                    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
                    sns.histplot(df[col_choice].dropna(), kde=True, ax=ax, color="#0D9488")
                    ax.set_title(f"Distribution of {col_choice}", fontsize=11)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=False)

            if len(numeric_cols) >= 2:
                st.write("")
                with st.container(border=True):
                    st.markdown('<div class="panel-title">Correlation heatmap</div>', unsafe_allow_html=True)
                    corr = correlation_matrix(df)
                    n = len(corr.columns)
                    size = max(6, min(0.55 * n, 12))
                    show_annot = n <= 9
                    fig2, ax2 = plt.subplots(figsize=(size, size * 0.85), dpi=120)
                    sns.heatmap(
                        corr, annot=show_annot, fmt=".2f", cmap="coolwarm",
                        ax=ax2, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8},
                        square=True, linewidths=0.5,
                    )
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
                    fig2.tight_layout()
                    st.pyplot(fig2, use_container_width=False)
                    if not show_annot:
                        st.caption("Cell values hidden above 9 columns to keep the heatmap legible — "
                                   "use the color scale, or narrow to fewer columns.")
        else:
            st.info("No numeric columns available for charting.")

        if cat_cols:
            st.write("")
            with st.container(border=True):
                st.markdown('<div class="panel-title">Category counts</div>', unsafe_allow_html=True)
                cat_choice = st.selectbox("Column", cat_cols, label_visibility="collapsed", key="cat_col")
                chart_col2, _ = st.columns([2, 1])
                with chart_col2:
                    fig3, ax3 = plt.subplots(figsize=(6, 3.5), dpi=120)
                    df[cat_choice].value_counts().head(15).plot(kind="bar", ax=ax3, color="#1E40AF")
                    ax3.set_title(f"Top values in {cat_choice}", fontsize=11)
                    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                    fig3.tight_layout()
                    st.pyplot(fig3, use_container_width=False)

    # -------------------------------------------------------
    # TAB 3: AI Insights (Qwen via Ollama)
    # -------------------------------------------------------
    with tab3:
        with st.container(border=True):
            st.markdown('<div class="panel-title">AI-generated insights</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">A structured summary of your data, generated on-device</div>',
                        unsafe_allow_html=True)

            if st.button("Generate insights"):
                if not ollama_ok:
                    st.error("The AI engine isn't running. Start it and refresh this page.")
                else:
                    prompt = f"""You are a senior data analyst. Analyze the dataset summary below
and give clear, practical findings.

{context_summary}

Give your answer in this format:
1. Key Observations (bullet points)
2. Data Quality Issues (missing values, duplicates, outliers if any)
3. 3 Actionable Insights
4. 1 Recommendation for next steps

Be concise and specific to the actual columns and numbers shown above."""
                    with st.spinner("Analyzing locally..."):
                        try:
                            result = ask_ollama(prompt, model=model_name)
                            st.markdown(result)
                        except RuntimeError as e:
                            st.error(str(e))

    # -------------------------------------------------------
    # TAB 4: Q&A Chat
    # -------------------------------------------------------
    with tab4:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Ask AI about your data</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">Computes directly on your dataset, then explains the result in '
                        'plain language.</div>', unsafe_allow_html=True)

            for role, msg in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(msg)

            user_q = st.chat_input("Ask something about your data...")
            if user_q:
                st.session_state.chat_history.append(("user", user_q))
                with st.chat_message("user"):
                    st.markdown(user_q)

                if not ollama_ok:
                    answer = "The AI engine isn't running. Start it and refresh this page."
                else:
                    with st.spinner("Working on it..."):
                        answer = answer_question(user_q, df, context_summary, model=model_name)

                st.session_state.chat_history.append(("assistant", answer))
                with st.chat_message("assistant"):
                    st.markdown(answer)

else:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Get started</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-sub">Upload a dataset from the sidebar to begin.</div>
        <ol style="color:#374151; font-size:0.92rem; line-height:1.7;">
            <li><b>Upload</b> a CSV or Excel file from the sidebar</li>
            <li>Your data is profiled — shape, structure, missing values, summary statistics</li>
            <li>That profile is analyzed by the <b>on-device AI engine</b> — no API key, ₹0 cost</li>
            <li>Explore <b>Overview</b>, <b>Visual Analysis</b>, <b>AI Insights</b>, and <b>Ask AI</b> — entirely offline</li>
        </ol>
        """, unsafe_allow_html=True)

st.markdown('<div class="footnote">Runs 100% locally — no data leaves this machine.</div>',
            unsafe_allow_html=True)