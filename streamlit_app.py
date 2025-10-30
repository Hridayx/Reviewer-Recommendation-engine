import streamlit as st
import tempfile
from pathlib import Path
import importlib.util
import pandas as pd
import io
import traceback


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_data(show_spinner=False)
def run_rerank_pipeline(pdf_path: str, top_k: int = 10):
    repo_root = Path(__file__).resolve().parent
    rerank_file = repo_root / "Re-Ranking.py"
    if not rerank_file.exists():
        raise FileNotFoundError(f"Could not find {rerank_file}")

    mod = load_module_from_path(rerank_file, "re_ranking_module")

    if not hasattr(mod, "get_reranked_recommendations"):
        raise AttributeError("Module does not expose get_reranked_recommendations(pdf_path, top_k)")

    results = mod.get_reranked_recommendations(pdf_path, top_k=top_k)

    df = pd.DataFrame(results)
    if not df.empty and 'boosts' in df.columns:
        boosts_df = pd.json_normalize(df['boosts']).add_prefix('boost_')
        df = pd.concat([df.drop(columns=['boosts']), boosts_df], axis=1)

    return results, df


def main():
    st.set_page_config(
        page_title="Reviewer Recommendation",
        layout="wide",
        page_icon="✨"
    )

    # --- 🎨 Enhanced UI Styling ---
        # --- 🎨 Enhanced UI Styling (Dark + Light Mode Compatible) ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
        font-size: 20px;
        line-height: 1.7;
    }

    /* Detect dark mode dynamically */
    @media (prefers-color-scheme: dark) {
        html, body, [class*="css"] {
            background-color: #0e1117 !important;
            color: #f0f2f6 !important;
        }
        [data-testid="stExpander"] {
            background: #1a1d25 !important;
            border: 1px solid #2b7cff33 !important;
            color: #e4e6eb !important;
        }
        .stDataFrame table {
            color: #e4e6eb !important;
        }
    }

    @media (prefers-color-scheme: light) {
        html, body, [class*="css"] {
            background-color: #f7faff !important;
            color: #1c1c1e !important;
        }
        [data-testid="stExpander"] {
            background: #f9fbff !important;
            border: 1px solid #dbe5ff !important;
            color: #1c1c1e !important;
        }
    }

    /* Expander hover effect */
    [data-testid="stExpander"]:hover {
        border-color: #2b7cff !important;
        box-shadow: 0 0 10px rgba(43,124,255,0.2);
        transition: all 0.3s ease-in-out;
    }

    /* Title */
    /* Title */
    .stApp h1 {
        font-size: 75px !important;      /* 🔥 Bigger title */
        font-weight: 800 !important;     /* Extra bold */
        text-align: center !important;
        color: #2b7cff !important;       /* Accent blue */
        margin-bottom: 10px !important;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px rgba(43, 124, 255, 0.25);  /* Soft glow for depth */
    }


    .stButton>button {
        background: linear-gradient(90deg, #2b7cff, #1d5df5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 26px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(43, 124, 255, 0.2);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 15px rgba(43, 124, 255, 0.3);
    }

    /* Table font size */
    .stDataFrame table {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)


    # --- 🌟 Title & Description ---
    st.markdown("<h1>Reviewer Recommendation </h1>", unsafe_allow_html=True)
    st.markdown("<p>Upload a research paper (PDF) or enter a path to get the top reviewer recommendations.</p>", unsafe_allow_html=True)

    # --- 📥 Input Section ---
    st.markdown("<hr style='margin-top:15px;margin-bottom:25px;'>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("📄 Upload Paper (PDF)", type=['pdf'])
        manual_path = st.text_input("📂 Or enter a local PDF path (leave blank if uploading)")

    with col2:
        top_k = st.number_input("🔢 Top K Reviewers", min_value=1, max_value=100, value=10, help="Number of top reviewers to display")
        run_button = st.button("🚀 Start Ranking")

    # --- ⚙️ Execution ---
    if run_button:
        try:
            if uploaded is None and not manual_path:
                st.warning("⚠️ Please upload a PDF or provide a local path before running.")
                return

            if uploaded is not None:
                tmpdir = tempfile.mkdtemp(prefix="rr_engine_")
                tmp_path = Path(tmpdir) / uploaded.name
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                pdf_path = str(tmp_path)
            else:
                pdf_path = manual_path

            with st.spinner("⏳ Running re-ranking pipeline (this may take a while)..."):
                results, df = run_rerank_pipeline(pdf_path, top_k=top_k)

            if not results:
                st.info("ℹ️ No results returned from the pipeline.")
                return

            st.success(f"✅ Completed — Top {len(results)} reviewer results computed")

            # --- 🧩 Tiered Results ---
            tiers = df['tier'].unique() if 'tier' in df.columns else []
            for tier in tiers:
                st.markdown(f"<h3 style='color:#2b7cff;'> Tier: {tier}</h3>", unsafe_allow_html=True)
                sub_df = df[df['tier'] == tier].copy()
                display_cols = [c for c in ['rank', 'author', 'score', 'num_papers', 'institution', 'avg_similarity_pct'] if c in sub_df.columns]
                if display_cols:
                    st.dataframe(sub_df[display_cols].reset_index(drop=True), use_container_width=True)

                for _, row in sub_df.iterrows():
                    with st.expander(f"🔍 {int(row.get('rank', 0))}. {row.get('author', 'Unknown')}"):
                        st.json(row.to_dict())

            # --- 📊 Full Table ---
            st.markdown("<h3 style='color:#1d5df5;'>📈 Full Results Table</h3>", unsafe_allow_html=True)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

            # --- 💾 Download ---
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Results as CSV",
                csv,
                file_name="reranked_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("❌ An error occurred while running the pipeline.")
            st.exception(e)


if __name__ == "__main__":
    main()
