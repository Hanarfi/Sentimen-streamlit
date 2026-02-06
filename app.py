import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import io
import csv

import nltk
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# THEME (Dark Modern) - CSS
# =========================================================
def inject_dark_theme():
    st.markdown(
        """
        <style>
            /* Page background */
            .stApp {
                background: radial-gradient(1200px 700px at 10% 10%, #111827 0%, #0b1020 40%, #050814 100%);
                color: #E5E7EB;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b1020 0%, #060a17 100%);
                border-right: 1px solid rgba(255,255,255,0.06);
            }

            /* Headings */
            h1, h2, h3, h4 {
                color: #F9FAFB !important;
                letter-spacing: 0.2px;
            }

            /* Text */
            p, li, label, div, span {
                color: #E5E7EB;
            }

            /* Buttons */
            .stButton>button {
                background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%);
                color: white;
                border: 0;
                border-radius: 12px;
                padding: 0.6rem 1rem;
                font-weight: 600;
                box-shadow: 0 10px 30px rgba(0,0,0,0.35);
                transition: transform 0.06s ease-in-out;
            }
            .stButton>button:hover {
                transform: translateY(-1px);
                filter: brightness(1.05);
            }

            /* Secondary buttons (download etc.) */
            .stDownloadButton>button {
                background: linear-gradient(90deg, #10B981 0%, #22C55E 100%);
                color: #052e1a;
                border: 0;
                border-radius: 12px;
                padding: 0.6rem 1rem;
                font-weight: 700;
            }

            /* Dataframe container */
            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px;
                overflow: hidden;
            }

            /* Cards (custom) */
            .card {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 16px 18px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.25);
            }
            .muted {
                color: rgba(229,231,235,0.75);
                font-size: 0.95rem;
            }
            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(37,99,235,0.18);
                border: 1px solid rgba(37,99,235,0.25);
                color: #BFDBFE;
                font-weight: 700;
                font-size: 0.85rem;
            }

            /* Inputs */
            .stTextInput input, .stSelectbox div, .stFileUploader div {
                border-radius: 12px !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# NLTK setup (Cloud-friendly)
# =========================================================
@st.cache_resource
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

ensure_nltk()


# =========================================================
# Load resources from repo (no user input needed)
# =========================================================
KAMUS_PATH = "kamuskatabaku (1).xlsx"
LEX_POS_PATH = "positive.csv"
LEX_NEG_PATH = "negative.csv"


@st.cache_resource
def load_kamus_repo(path: str) -> dict:
    df = pd.read_excel(path)
    # expects columns: non_standard, standard_word
    return dict(zip(df["non_standard"], df["standard_word"]))


@st.cache_resource
def load_lexicon_repo(path: str) -> dict:
    lex = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if len(row) >= 2:
                try:
                    lex[row[0]] = int(row[1])
                except:
                    pass
    return lex


def safe_load_resources():
    errors = []
    kamus = None
    lex_pos = None
    lex_neg = None

    try:
        kamus = load_kamus_repo(KAMUS_PATH)
    except Exception as e:
        errors.append(f"Gagal load kamus: {e}")

    try:
        lex_pos = load_lexicon_repo(LEX_POS_PATH)
    except Exception as e:
        errors.append(f"Gagal load lexicon positive: {e}")

    try:
        lex_neg = load_lexicon_repo(LEX_NEG_PATH)
    except Exception as e:
        errors.append(f"Gagal load lexicon negative: {e}")

    return kamus, lex_pos, lex_neg, errors


# =========================================================
# Preprocessing functions (stabil, sesuai revisi)
# =========================================================
def CaseFolding(text: str) -> str:
    return str(text).lower()

def normalisasi_dengan_kamus(text: str, kamus_dict: dict) -> str:
    words = str(text).split()
    normalized = [kamus_dict.get(w, w) for w in words]
    return " ".join(normalized)

def datacleaning(text: str) -> str:
    text = str(text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"RT[\s]", "", text)
    text = re.sub(r"RT[?|$|.|@!&:_=)(><,]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]", "", text)
    text = text.replace("\n", " ")
    text = text.strip(" ")
    text = re.sub("s://t.co/", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.replace('"', "")
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words("indonesian"))
    tokens = str(text).split()   # <- aman (tidak perlu punkt)
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

@st.cache_resource
def get_sastrawi_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

def stem_text(text: str) -> str:
    stemmer = get_sastrawi_stemmer()
    return stemmer.stem(str(text))

def filter_tokens_by_lexicon(tokens, lex_pos: dict, lex_neg: dict):
    allowed = set(lex_pos.keys()) | set(lex_neg.keys())
    return [t for t in (tokens or []) if t in allowed]


# =========================================================
# Labeling Lexicon
# =========================================================
def sentiment_analysis_lexicon_indonesia(tokens, lex_pos: dict, lex_neg: dict):
    score = 0
    for w in tokens:
        if w in lex_pos:
            score += lex_pos[w]
    for w in tokens:
        if w in lex_neg:
            score += lex_neg[w]

    if score > 0:
        sent = "positif"
    elif score < 0:
        sent = "negatif"
    else:
        sent = "netral"
    return score, sent


# =========================================================
# Helpers
# =========================================================
def show_preview(df: pd.DataFrame, title: str, n=20):
    st.markdown(f"#### {title}")
    st.dataframe(df.head(n), use_container_width=True)

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["content"] = out["content"].fillna("").astype(str)
    out = out[out["content"].str.strip() != ""]
    out = out.reset_index(drop=True)
    return out

def to_excel_bytes(df: pd.DataFrame, sheet_name="data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()

def plot_confusion(cm, labels=("negatif", "positif"), title="Confusion Matrix"):
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(labels), yticklabels=list(labels))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# Session State
# =========================================================
def init_state():
    defaults = {
        "menu": "Home",
        "raw_df": None,
        "df_work": None,
        "chosen_col": None,

        # resources
        "kamus": None,
        "lex_pos": None,
        "lex_neg": None,
        "res_errors": [],

        # preprocessing checkpoints
        "pp_casefold": None,
        "pp_normal": None,
        "pp_clean": None,
        "pp_stop": None,
        "pp_stem": None,
        "pp_filterlex": None,
        "pp_labeled": None,

        # model artifacts
        "tfidf": None,
        "tfidf_df": None,
        "X_tfidf": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "svm": None,
        "y_pred": None,
        "report": None,
        "cm": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# =========================================================
# Page config & theme
# =========================================================
st.set_page_config(page_title="Sentiment Analysis (TF-IDF + SVM)", layout="wide")
inject_dark_theme()

# Load resources once (kamus & lexicon from repo)
if st.session_state.kamus is None or st.session_state.lex_pos is None or st.session_state.lex_neg is None:
    kamus, lex_pos, lex_neg, errs = safe_load_resources()
    st.session_state.kamus = kamus
    st.session_state.lex_pos = lex_pos
    st.session_state.lex_neg = lex_neg
    st.session_state.res_errors = errs


# =========================================================
# Sidebar Navigation
# =========================================================
st.sidebar.markdown("## 📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Home", "Input", "Preprocessing", "Klasifikasi SVM"],
    index=["Home", "Input", "Preprocessing", "Klasifikasi SVM"].index(st.session_state.menu)
)
st.session_state.menu = menu

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Resource (Repo)")
if st.session_state.res_errors:
    st.sidebar.error("Resource gagal dimuat.")
    for e in st.session_state.res_errors:
        st.sidebar.write(f"- {e}")
else:
    st.sidebar.success("Kamus & Lexicon siap.")
    st.sidebar.write(f"- Kamus: {len(st.session_state.kamus)} entri")
    st.sidebar.write(f"- Lexicon +: {len(st.session_state.lex_pos)} entri")
    st.sidebar.write(f"- Lexicon -: {len(st.session_state.lex_neg)} entri")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Parameter SVM")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)
C = st.sidebar.number_input("SVM C", min_value=0.01, value=1.0, step=0.1)
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], index=0)


# =========================================================
# HOME
# =========================================================
if st.session_state.menu == "Home":
    st.markdown(
        """
        <div class="card">
            <span class="badge">Sistem Analisis Sentimen</span>
            <h2 style="margin-top:10px;">TF-IDF + SVM untuk Ulasan Pengguna</h2>
            <p class="muted">
                Sistem ini melakukan analisis sentimen pada ulasan pengguna menggunakan tahapan:
                <b>Preprocessing</b> (case folding, normalisasi kamus, cleaning, stopword, stemming, filter lexicon),
                kemudian <b>Labeling Lexicon</b> (positif/negatif/netral).
                Setelah itu data diproses dengan <b>TF-IDF</b> dan diklasifikasikan menggunakan <b>SVM</b>.
            </p>
            <p class="muted">
                Kamu bisa menjalankan proses secara bertahap dan melihat hasil di setiap tahap.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        if st.button("🚀 Mulai"):
            st.session_state.menu = "Input"
            st.rerun()
    with colB:
        if st.button("🧹 Reset Semua"):
            for k in list(st.session_state.keys()):
                if k not in ["menu", "kamus", "lex_pos", "lex_neg", "res_errors"]:
                    st.session_state[k] = None
            st.session_state.menu = "Home"
            st.rerun()


# =========================================================
# INPUT
# =========================================================
elif st.session_state.menu == "Input":
    st.markdown(
        """
        <div class="card">
            <h2>📥 Input Data Ulasan</h2>
            <p class="muted">
                Upload file CSV ulasan. Nama kolom bebas — setelah upload kamu bisa memilih kolom yang berisi teks ulasan.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=",", skipinitialspace=True, na_values="?")
        st.session_state.raw_df = df.copy()

        show_preview(df, "Preview Dataset (Raw)", n=20)

        st.markdown("#### Pilih Kolom Teks Ulasan")
        chosen_col = st.selectbox("Kolom teks", options=df.columns.tolist())
        st.session_state.chosen_col = chosen_col

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("✅ Gunakan Kolom Ini"):
                work = pd.DataFrame()
                work["content"] = df[chosen_col].astype(str)
                work = drop_empty_rows(work)

                # reset hasil-hasil sebelumnya
                st.session_state.df_work = work
                st.session_state.pp_casefold = None
                st.session_state.pp_normal = None
                st.session_state.pp_clean = None
                st.session_state.pp_stop = None
                st.session_state.pp_stem = None
                st.session_state.pp_filterlex = None
                st.session_state.pp_labeled = None
                st.session_state.tfidf = None
                st.session_state.tfidf_df = None
                st.session_state.X_tfidf = None
                st.session_state.svm = None
                st.session_state.report = None
                st.session_state.cm = None

                st.success(f"Kolom '{chosen_col}' dipakai sebagai ulasan. Data siap untuk preprocessing.")
        with col2:
            st.info("Lanjut ke menu **Preprocessing** lewat sidebar.")

    if st.session_state.df_work is not None:
        st.write("")
        show_preview(st.session_state.df_work, "Data yang Akan Diproses", n=20)


# =========================================================
# PREPROCESSING (awam-friendly, step-by-step, preview tiap tahap + download excel)
# =========================================================
elif st.session_state.menu == "Preprocessing":
    st.markdown(
        """
        <div class="card">
            <h2>🧽 Preprocessing</h2>
            <p class="muted">
                Jalankan tahapan preprocessing secara bertahap. Setiap tahap akan menampilkan hasilnya agar mudah dipahami.
                Kamu juga bisa menjalankan semuanya sekaligus.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    if st.session_state.df_work is None:
        st.warning("Belum ada data. Silakan upload CSV pada menu **Input**.")
    elif st.session_state.res_errors:
        st.error("Resource (kamus/lexicon) gagal dimuat. Pastikan file ada di repo dan namanya sesuai.")
    else:
        base_df = st.session_state.df_work.copy()

        # Action buttons
        col_run1, col_run2, col_run3 = st.columns([1.2, 1.2, 2])
        with col_run1:
            run_all = st.button("▶️ Jalankan Semua Preprocessing")
        with col_run2:
            reset_pp = st.button("🔄 Reset Preprocessing")
        with col_run3:
            st.caption("Tips: Jalankan dari atas ke bawah agar hasilnya konsisten.")

        if reset_pp:
            st.session_state.pp_casefold = None
            st.session_state.pp_normal = None
            st.session_state.pp_clean = None
            st.session_state.pp_stop = None
            st.session_state.pp_stem = None
            st.session_state.pp_filterlex = None
            st.session_state.pp_labeled = None
            st.success("Preprocessing direset.")
            st.rerun()

        # Define pipeline steps
        def step_casefold(df):
            out = df.copy()
            out["content"] = out["content"].apply(CaseFolding)
            return drop_empty_rows(out)

        def step_normalisasi(df):
            out = df.copy()
            out["content"] = out["content"].apply(lambda x: normalisasi_dengan_kamus(x, st.session_state.kamus))
            return drop_empty_rows(out)

        def step_clean(df):
            out = df.copy()
            out["content"] = out["content"].apply(datacleaning)
            return drop_empty_rows(out)

        def step_stopword(df):
            out = df.copy()
            out["content"] = out["content"].apply(remove_stopwords)
            return drop_empty_rows(out)

        def step_stemming(df):
            out = df.copy()
            out["content"] = out["content"].apply(stem_text)
            return drop_empty_rows(out)

        def step_filterlex(df):
            out = df.copy()
            out["content_list"] = out["content"].astype(str).str.split()
            out["content_list"] = out["content_list"].apply(
                lambda toks: filter_tokens_by_lexicon(toks, st.session_state.lex_pos, st.session_state.lex_neg)
            )
            out["content"] = out["content_list"].apply(lambda toks: " ".join(toks))
            out = drop_empty_rows(out)
            return out

        def step_labeling(df):
            out = df.copy()
            if "content_list" not in out.columns:
                out["content_list"] = out["content"].astype(str).str.split()
            res = out["content_list"].apply(lambda toks: sentiment_analysis_lexicon_indonesia(
                toks, st.session_state.lex_pos, st.session_state.lex_neg
            ))
            res = list(zip(*res))
            out["score"] = res[0]
            out["Sentimen"] = res[1]
            return out

        # Run all
        if run_all:
            st.session_state.pp_casefold = step_casefold(base_df)
            st.session_state.pp_normal = step_normalisasi(st.session_state.pp_casefold)
            st.session_state.pp_clean = step_clean(st.session_state.pp_normal)
            st.session_state.pp_stop = step_stopword(st.session_state.pp_clean)
            st.session_state.pp_stem = step_stemming(st.session_state.pp_stop)
            st.session_state.pp_filterlex = step_filterlex(st.session_state.pp_stem)
            st.session_state.pp_labeled = step_labeling(st.session_state.pp_filterlex)
            st.success("Semua tahap preprocessing selesai. Lihat hasil di bawah.")

        # UI step-by-step (simple for awam) using expanders
        st.markdown("### Tahapan Preprocessing")

        # STEP 1
        with st.expander("1) Case Folding (huruf kecil semua)", expanded=True):
            if st.button("Jalankan Case Folding"):
                st.session_state.pp_casefold = step_casefold(base_df)
            if st.session_state.pp_casefold is not None:
                show_preview(st.session_state.pp_casefold, "Hasil Case Folding", n=20)

        # STEP 2
        with st.expander("2) Normalisasi Kamus (kata tidak baku → baku)", expanded=False):
            if st.button("Jalankan Normalisasi Kamus"):
                prev = st.session_state.pp_casefold if st.session_state.pp_casefold is not None else base_df
                st.session_state.pp_normal = step_normalisasi(prev)
            if st.session_state.pp_normal is not None:
                show_preview(st.session_state.pp_normal, "Hasil Normalisasi Kamus", n=20)

        # STEP 3
        with st.expander("3) Data Cleaning (hapus simbol/angka/link/emoji)", expanded=False):
            if st.button("Jalankan Data Cleaning"):
                prev = st.session_state.pp_normal if st.session_state.pp_normal is not None else (
                    st.session_state.pp_casefold if st.session_state.pp_casefold is not None else base_df
                )
                st.session_state.pp_clean = step_clean(prev)
            if st.session_state.pp_clean is not None:
                show_preview(st.session_state.pp_clean, "Hasil Data Cleaning", n=20)

        # STEP 4
        with st.expander("4) Stopword Removal (hapus kata umum)", expanded=False):
            if st.button("Jalankan Stopword Removal"):
                prev = st.session_state.pp_clean if st.session_state.pp_clean is not None else (
                    st.session_state.pp_normal if st.session_state.pp_normal is not None else base_df
                )
                st.session_state.pp_stop = step_stopword(prev)
            if st.session_state.pp_stop is not None:
                show_preview(st.session_state.pp_stop, "Hasil Stopword Removal", n=20)

        # STEP 5
        with st.expander("5) Stemming (ubah kata ke bentuk dasar)", expanded=False):
            if st.button("Jalankan Stemming"):
                prev = st.session_state.pp_stop if st.session_state.pp_stop is not None else (
                    st.session_state.pp_clean if st.session_state.pp_clean is not None else base_df
                )
                st.session_state.pp_stem = step_stemming(prev)
            if st.session_state.pp_stem is not None:
                show_preview(st.session_state.pp_stem, "Hasil Stemming", n=20)

        # STEP 6
        with st.expander("6) Filter Lexicon (hapus kata typo/OOV yang tidak ada di lexicon)", expanded=False):
            if st.button("Jalankan Filter Lexicon"):
                prev = st.session_state.pp_stem if st.session_state.pp_stem is not None else (
                    st.session_state.pp_stop if st.session_state.pp_stop is not None else base_df
                )
                st.session_state.pp_filterlex = step_filterlex(prev)
            if st.session_state.pp_filterlex is not None:
                show_preview(st.session_state.pp_filterlex, "Hasil Filter Lexicon", n=20)

        # STEP 7 (Labeling)
        with st.expander("7) Labeling Lexicon (positif/negatif/netral + score)", expanded=False):
            if st.button("Jalankan Labeling Lexicon"):
                prev = st.session_state.pp_filterlex if st.session_state.pp_filterlex is not None else (
                    st.session_state.pp_stem if st.session_state.pp_stem is not None else base_df
                )
                st.session_state.pp_labeled = step_labeling(prev)

            if st.session_state.pp_labeled is not None:
                df_lab = st.session_state.pp_labeled
                show_preview(df_lab, "Hasil Labeling", n=20)
                st.write("Distribusi Sentimen:")
                st.write(df_lab["Sentimen"].value_counts())

                col_f1, col_f2 = st.columns([1, 2])
                with col_f1:
                    if st.button("Filter Netral (score == 0)"):
                        df2 = df_lab[df_lab["score"] != 0].reset_index(drop=True)
                        st.session_state.pp_labeled = df2
                        st.success("Netral dihapus.")
                with col_f2:
                    st.caption("SVM nanti memakai label negatif/positif (netral biasanya dibuang).")

        # Download at end of preprocessing
        st.markdown("---")
        st.markdown("### ⬇️ Download Hasil Preprocessing (Excel)")
        if st.session_state.pp_labeled is None:
            st.info("Jalankan minimal sampai tahap Labeling dulu agar hasil lengkap.")
        else:
            excel_bytes = to_excel_bytes(st.session_state.pp_labeled, sheet_name="preprocessing")
            st.download_button(
                "Download Excel Preprocessing",
                data=excel_bytes,
                file_name="hasil_preprocessing.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# =========================================================
# KLASIFIKASI SVM (TF-IDF -> tampil lengkap -> Split -> SVM -> report + CM -> download excel)
# =========================================================
elif st.session_state.menu == "Klasifikasi SVM":
    st.markdown(
        """
        <div class="card">
            <h2>🤖 Klasifikasi SVM</h2>
            <p class="muted">
                Jalankan tahapan model secara bertahap: <b>TF-IDF</b> → <b>Split Data</b> → <b>Klasifikasi SVM</b> →
                <b>Classification Report</b> dan <b>Confusion Matrix</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    if st.session_state.pp_labeled is None:
        st.warning("Data belum melalui preprocessing + labeling. Silakan ke menu **Preprocessing** dan jalankan sampai Labeling.")
    else:
        df = st.session_state.pp_labeled.copy()

        # Pastikan hanya negatif/positif (umumnya netral dibuang)
        if "Sentimen" not in df.columns:
            st.error("Kolom 'Sentimen' belum ada. Jalankan labeling di menu Preprocessing.")
        else:
            df = df[df["Sentimen"].isin(["negatif", "positif"])].reset_index(drop=True)
            if df.empty:
                st.error("Data negatif/positif kosong. Pastikan ada data setelah filter netral.")
            else:
                if "content_list" not in df.columns:
                    df["content_list"] = df["content"].astype(str).str.split()

                # BUTTONS
                b1, b2, b3 = st.columns([1.1, 1.1, 2])
                with b1:
                    run_tfidf = st.button("1) Jalankan TF-IDF")
                with b2:
                    run_split = st.button("2) Split Data")
                with b3:
                    run_svm = st.button("3) Klasifikasi SVM")

                # 1) TF-IDF
                if run_tfidf or (st.session_state.tfidf is not None and st.session_state.tfidf_df is None):
                    X_text = df["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
                    tfidf = TfidfVectorizer()
                    X_tfidf = tfidf.fit_transform(X_text).toarray()

                    tfidf_df = pd.DataFrame(X_tfidf, columns=tfidf.get_feature_names_out())
                    st.session_state.tfidf = tfidf
                    st.session_state.X_tfidf = X_tfidf
                    st.session_state.tfidf_df = tfidf_df

                    st.success(f"TF-IDF selesai. Total fitur: {tfidf_df.shape[1]}")

                if st.session_state.tfidf_df is not None:
                    st.markdown("### Hasil TF-IDF (Lengkap)")
                    st.caption("Catatan: Jika dataset sangat besar, menampilkan semua fitur bisa berat. Namun tabel di bawah adalah hasil lengkap (scroll).")
                    st.dataframe(st.session_state.tfidf_df, use_container_width=True)

                    tfidf_excel = to_excel_bytes(st.session_state.tfidf_df, sheet_name="tfidf")
                    st.download_button(
                        "Download TF-IDF (Excel)",
                        data=tfidf_excel,
                        file_name="hasil_tfidf.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # 2) Split
                if run_split:
                    if st.session_state.X_tfidf is None:
                        st.error("Jalankan TF-IDF dulu.")
                    else:
                        y = df["Sentimen"]
                        X_train, X_test, y_train, y_test = train_test_split(
                            st.session_state.X_tfidf, y,
                            test_size=float(test_size),
                            random_state=int(random_state)
                        )
                        st.session_state.X_train, st.session_state.X_test = X_train, X_test
                        st.session_state.y_train, st.session_state.y_test = y_train, y_test
                        st.success(f"Split data selesai. Train: {len(y_train)} | Test: {len(y_test)}")

                # 3) SVM
                if run_svm:
                    if st.session_state.X_train is None or st.session_state.X_test is None:
                        st.error("Jalankan Split Data dulu.")
                    else:
                        svm = SVC(kernel=kernel, C=float(C))
                        svm.fit(st.session_state.X_train, st.session_state.y_train)
                        y_pred = svm.predict(st.session_state.X_test)

                        st.session_state.svm = svm
                        st.session_state.y_pred = y_pred

                        acc = accuracy_score(st.session_state.y_test, y_pred)
                        st.success(f"Klasifikasi SVM selesai. Accuracy: {acc:.4f}")

                        st.session_state.report = classification_report(
                            st.session_state.y_test, y_pred, zero_division=0
                        )
                        st.session_state.cm = confusion_matrix(
                            st.session_state.y_test, y_pred, labels=["negatif", "positif"]
                        )

                # Results
                if st.session_state.report is not None:
                    st.markdown("### Classification Report")
                    st.code(st.session_state.report)

                if st.session_state.cm is not None:
                    st.markdown("### Confusion Matrix")
                    plot_confusion(st.session_state.cm, labels=("negatif", "positif"), title="Confusion Matrix SVM")

                # Download final (df + predictions on ALL df, optional)
                st.markdown("---")
                st.markdown("### ⬇️ Download Hasil Klasifikasi (Excel)")

                if st.session_state.svm is None or st.session_state.tfidf is None:
                    st.info("Jalankan TF-IDF + Split Data + Klasifikasi SVM dulu.")
                else:
                    # Predict for all processed rows
                    X_all = st.session_state.tfidf.transform(
                        df["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
                    ).toarray()
                    df_out = df.copy()
                    df_out["Prediksi_SVM"] = st.session_state.svm.predict(X_all)

                    excel_bytes = to_excel_bytes(df_out, sheet_name="svm_results")
                    st.download_button(
                        "Download Excel Hasil Klasifikasi",
                        data=excel_bytes,
                        file_name="hasil_klasifikasi_svm.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
