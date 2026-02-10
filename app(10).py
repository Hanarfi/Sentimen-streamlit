
import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import io
import csv
import pickle
import time

from google_play_scraper import reviews, Sort

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
# UI STYLE (Dark, clean, academic)
# =========================================================
def inject_academic_dark():
    st.markdown(
        """
        <style>
            .stApp { background: #0b0f17; color: #E5E7EB; }
            section[data-testid="stSidebar"] {
                background: #0a0e16;
                border-right: 1px solid rgba(255,255,255,0.06);
            }
            h1, h2, h3, h4 { color: #F9FAFB !important; }
            p, li, label, div, span { color: #E5E7EB; }

            .stButton>button {
                background: #1f2937;
                color: #F9FAFB;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 10px;
                padding: 0.55rem 0.9rem;
                font-weight: 650;
            }
            .stButton>button:hover {
                border-color: rgba(255,255,255,0.20);
                filter: brightness(1.05);
            }
            /* Segmented toggle buttons */
            .seg-wrap { display: flex; gap: 10px; margin: 6px 0 10px 0; }
            .seg-btn button {
                width: 100%;
                background: rgba(255,255,255,0.03);
                color: #E5E7EB;
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
                padding: 0.65rem 0.9rem;
                font-weight: 750;
            }
            .seg-btn button:hover {
                border-color: rgba(255,255,255,0.25);
                filter: brightness(1.08);
            }
            .seg-active button {
                background: linear-gradient(180deg, rgba(59,130,246,0.22), rgba(59,130,246,0.10));
                border: 1px solid rgba(59,130,246,0.45);
                box-shadow: 0 0 0 1px rgba(59,130,246,0.15) inset;
            }
            .seg-sub { color: rgba(229,231,235,0.65); font-size: 0.92rem; margin-top: -2px; }

            .stDownloadButton>button {
                background: #111827;
                color: #F9FAFB;
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 10px;
                padding: 0.55rem 0.9rem;
                font-weight: 700;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 10px;
                overflow: hidden;
            }

            hr {
                border: none;
                border-top: 1px solid rgba(255,255,255,0.08);
                margin: 1rem 0;
            }

            .muted { color: rgba(229,231,235,0.75); }
            .hint  { color: rgba(229,231,235,0.65); font-size: 0.92rem; }
            .kpi   {
                padding: 10px 12px; background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
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
# Resources from repo (no user input)
# =========================================================
KAMUS_PATH = "kamuskatabaku (1).xlsx"
LEX_POS_PATH = "positive.csv"
LEX_NEG_PATH = "negative.csv"

@st.cache_resource
def load_kamus_repo(path: str) -> dict:
    df = pd.read_excel(path)
    return dict(zip(df["non_standard"], df["standard_word"]))

@st.cache_resource
def load_lexicon_repo(path: str) -> dict:
    """
    Robust lexicon loader:
    - skip header jika ada
    - lower() word
    - nilai non-int di-skip
    """
    lex = {}
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        first = next(reader, None)
        # Coba deteksi header (kalau baris pertama bukan angka di kolom 2)
        if first and len(first) >= 2:
            w = str(first[1]).strip()
            try:
                int(w)
                # baris pertama valid data → proses juga
                word = str(first[0]).strip().lower()
                if word:
                    lex[word] = int(w)
            except ValueError:
                # anggap header → skip
                pass

        for row in reader:
            if len(row) >= 2:
                word = str(row[0]).strip().lower()
                w = str(row[1]).strip()
                if not word:
                    continue
                try:
                    lex[word] = int(w)
                except ValueError:
                    continue
    return lex


def safe_load_resources():
    errors = []
    kamus = lex_pos = lex_neg = None
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
# Preprocessing functions
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
    text = text.replace("\n", " ").strip(" ")
    text = re.sub("s://t.co/", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.replace('"', "")
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words("indonesian"))
    tokens = str(text).split()
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
    """
    Filter token berdasarkan lexicon positif & negatif
    """
    filtered_words = []

    if isinstance(tokens, list):
        for word in tokens:
            if word in lex_pos or word in lex_neg:
                filtered_words.append(word)

    return filtered_words


# =========================================================
# Labeling lexicon
# =========================================================
def sentiment_analysis_lexicon_indonesia(tokens):
    score = 0

    lex_single = st.session_state.get("lexicon_single", {}) or {}
    lex_phrase = st.session_state.get("lexicon_phrase", {}) or {}

    # 5a) score single-word tokens
    for w in tokens or []:
        score += lex_single.get(w, 0)

    # 5b) score phrase (bigram/trigram) ringan
    if lex_phrase and tokens:
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram in lex_phrase:
                score += lex_phrase[bigram]

        for i in range(len(tokens) - 2):
            trigram = tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2]
            if trigram in lex_phrase:
                score += lex_phrase[trigram]

    if score > 0:
        sentimen = "positif"
    elif score < 0:
        sentimen = "negatif"
    else:
        sentimen = "netral"

    return score, sentimen



# =========================================================
# Helpers
# =========================================================
def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["content"] = out["content"].fillna("").astype(str)
    out = out[out["content"].str.strip() != ""].reset_index(drop=True)
    return out

def show_preview(df: pd.DataFrame, title: str, n=20):
    st.subheader(title)
    st.dataframe(df.head(n), use_container_width=True)

def show_processed_count(df_before: pd.DataFrame, df_after: pd.DataFrame, title="Keterangan Jumlah Data"):
    n_before = 0 if df_before is None else len(df_before)
    n_after = 0 if df_after is None else len(df_after)
    dropped = max(n_before - n_after, 0)
    dropped_pct = (dropped / n_before * 100) if n_before else 0.0

    st.markdown(f"#### {title}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah data awal", f"{n_before}")
    c2.metric("Jumlah data terproses", f"{n_after}")
    c3.metric("Terbuang", f"{dropped} ({dropped_pct:.1f}%)")

def to_excel_bytes(df: pd.DataFrame, sheet_name="data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()

def plot_bar_counts(series: pd.Series, title: str):
    fig = plt.figure(figsize=(6, 4))
    counts = series.value_counts()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah")
    st.pyplot(fig)
    plt.close(fig)

def plot_confusion(cm, labels=("negatif", "positif"), title="Confusion Matrix"):
    cm = np.array(cm)
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(labels), yticklabels=list(labels))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.where(row_sums == 0, 0, cm / row_sums) * 100

    fig2 = plt.figure(figsize=(6, 4))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=list(labels), yticklabels=list(labels))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title + " (Persentase)")
    st.pyplot(fig2)
    plt.close(fig2)

def biggest_confusion_insight(cm, labels=("negatif", "positif")) -> str:
    cm = np.array(cm)
    if cm.shape != (2, 2):
        return "Insight error belum tersedia."
    fn = cm[0, 1]  # true negatif diprediksi positif
    fp = cm[1, 0]  # true positif diprediksi negatif
    if fn == 0 and fp == 0:
        return "Tidak ada kesalahan pada confusion matrix (data uji)."
    if fn >= fp:
        return f"Kesalahan terbesar: **{labels[0]} → {labels[1]}** sebanyak **{fn}**."
    return f"Kesalahan terbesar: **{labels[1]} → {labels[0]}** sebanyak **{fp}**."

def make_model_bundle(tfidf: TfidfVectorizer, svm: SVC):
    return {"tfidf": tfidf, "svm": svm}


# =========================================================
# Session state
# =========================================================
def init_state():
    defaults = {
        "menu": "Home",

        "mode_proses": "auto",  # "auto" atau "step"

        "raw_df": None,
        "df_work": None,
        "chosen_col": None,

        "kamus": None,
        "lex_pos": None,
        "lex_neg": None,
        "res_errors": [],

        # scraping artifacts
        "scraped_df": None,
        "scrape_meta": None,
        
        # preprocessing outputs
        "pp_casefold": None,
        "pp_normal": None,
        "pp_clean": None,
        "pp_stop": None,
        "pp_stem": None,
        "pp_token": None,
        "pp_filterlex": None,
        "pp_labeled": None,
        "pp_labeled_raw": None,

        # model artifacts
        "tfidf": None,
        "tfidf_df": None,
        "X_tfidf": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "svm": None,
        "y_pred": None,
        "report": None,
        "cm": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

st.set_page_config(page_title="Analisis Sentimen (TF-IDF + SVM)", layout="wide")
inject_academic_dark()

# load resources
if st.session_state.kamus is None or st.session_state.lex_pos is None or st.session_state.lex_neg is None:
    kamus, lex_pos, lex_neg, errs = safe_load_resources()
    st.session_state.kamus = kamus
    st.session_state.lex_pos = lex_pos
    st.session_state.lex_neg = lex_neg
    st.session_state.res_errors = errs

# ✅ lexicon gabungan (atasi overlap pos-neg) + pisah phrase/single
if not st.session_state.res_errors and st.session_state.lex_pos is not None and st.session_state.lex_neg is not None:
    from collections import defaultdict

    lex_all = defaultdict(int)
    for k, v in st.session_state.lex_pos.items():
        lex_all[str(k).strip().lower()] += int(v)
    for k, v in st.session_state.lex_neg.items():
        lex_all[str(k).strip().lower()] += int(v)

    lex_all = dict(lex_all)
    st.session_state.lexicon_all = lex_all
    st.session_state.lexicon_phrase = {k: v for k, v in lex_all.items() if " " in k}
    st.session_state.lexicon_single = {k: v for k, v in lex_all.items() if " " not in k}



# =========================================================
# Sidebar - navigation + progress + resources + reset
# =========================================================
def nav_button(label, icon, target_menu):
    is_active = (st.session_state.menu == target_menu)
    txt = f"{icon}  {label}"
    if is_active:
        st.sidebar.markdown(f"**➡️ {txt}**")
    else:
        if st.sidebar.button(txt):
            st.session_state.menu = target_menu
            st.rerun()

st.sidebar.markdown("## Navigasi")
nav_button("Home", "🏠", "Home")
nav_button("Scraping", "🕷️", "Scraping")
nav_button("Input", "📥", "Input")
nav_button("Proses", "🧽", "Proses")
nav_button("Klasifikasi SVM", "🤖", "Klasifikasi SVM")

st.sidebar.markdown("---")
st.sidebar.markdown("## Progress")
step_input = st.session_state.df_work is not None
step_labeled = st.session_state.pp_labeled is not None
step_tfidf = st.session_state.tfidf is not None and st.session_state.X_tfidf is not None
step_split = st.session_state.X_train is not None and st.session_state.X_test is not None
step_svm = st.session_state.svm is not None and st.session_state.y_pred is not None
st.sidebar.write(f"{'✅' if step_input else '⬜'} Input data")
st.sidebar.write(f"{'✅' if step_labeled else '⬜'} Preprocessing + labeling")
st.sidebar.write(f"{'✅' if step_tfidf else '⬜'} TF-IDF")
st.sidebar.write(f"{'✅' if step_split else '⬜'} Split data")
st.sidebar.write(f"{'✅' if step_svm else '⬜'} Klasifikasi SVM")

st.sidebar.markdown("---")
st.sidebar.markdown("## Resource")
if st.session_state.res_errors:
    st.sidebar.error("Resource gagal dimuat.")
    for e in st.session_state.res_errors:
        st.sidebar.write(f"- {e}")
else:
    st.sidebar.success("Resource siap.")
    st.sidebar.write(f"- Kamus: {len(st.session_state.kamus)}")
    st.sidebar.write(f"- Lexicon +: {len(st.session_state.lex_pos)}")
    st.sidebar.write(f"- Lexicon -: {len(st.session_state.lex_neg)}")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset"):
    keep = {"menu", "kamus", "lex_pos", "lex_neg", "res_errors"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            st.session_state[k] = None
    st.session_state.menu = "Home"
    st.rerun()


# =========================================================
# HOME
# =========================================================
if st.session_state.menu == "Home":
    st.title("Sistem Analisis Sentimen Ulasan Pengguna pada Aplikasi-aplikasi Google Play")
    st.write(
        "Metode yang digunakan: **Preprocessing + Labeling Lexicon → TF-IDF → SVM**. "
        "Aplikasi ini dirancang agar proses dapat dijalankan bertahap dan hasil tiap tahap dapat diamati."
    )

    st.markdown("### Alur Penggunaan")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("<div class='kpi'><b>1) 🕷️ Scraping (Opsional)</b><div class='hint'>Scraping dataset ulasan.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kpi'><b>2) 📥 Input</b><div class='hint'>Unggah CSV dan pilih kolom teks.</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kpi'><b>3) 🧽 Proses</b><div class='hint'>Pembersihan teks (otomatis / tahap).</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='kpi'><b>4) 🤖 Klasifikasi</b><div class='hint'>TF-IDF, split, SVM.</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown("<div class='kpi'><b>5) 📊 Hasil</b><div class='hint'>Report, CM, dan output.</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚀 Mulai"):
        st.session_state.menu = "Scraping"
        st.rerun()


# =========================================================
# SCRAPING (Google Play Reviews) - OPSIONAL + bisa langsung jadi dataset
# =========================================================
elif st.session_state.menu == "Scraping":
    st.title("Scraping Ulasan Google Play (Opsional)")
    st.write(
        "Menu ini **opsional**. Gunakan jika kamu belum punya dataset CSV. "
        "Jika sudah punya dataset, langsung ke menu **Input** untuk upload."
    )

    st.markdown("### Panduan Penggunaan")
    st.markdown(
        """
        1. Masukkan **Package Name** aplikasi (contoh: `co.id.bankbsi.superapp`).  
        2. Pilih **Sort** (Newest / Most Relevant).  
        3. Tentukan jumlah review (**Count**).  
        4. Klik **Ambil Reviews** → hasil tampil dan bisa:
           - **diunduh sebagai CSV**, atau
           - **langsung dipakai sebagai dataset** (tanpa upload ulang).  
        
        - **Cara mencari package name :** buka Google Play → URL biasanya mengandung `id=<package_name>`.
        - **Contohnya :** https://play.google.com/store/apps/details?id=co.id.bankbsi.superapp&hl=en.
        - Pada contoh diatas ambil pada bagian setelah id (abaikan "=" "&hl=en") yaitu co.id.bankbsi.superapp
        """
    )

    st.markdown("---")
    st.subheader("Parameter Scraping")

    col1, col2, col3 = st.columns(3)
    with col1:
        package_name = st.text_input("Package Name", value="co.id.bankbsi.superapp")
    with col2:
        sort_choice = st.selectbox("Sort", ["NEWEST", "MOST_RELEVANT"], index=0)
    with col3:
        count = st.number_input("Count (jumlah review)", min_value=10, max_value=500, value=50, step=10)

    lang = st.text_input("Language (lang)", value="id")
    country = st.text_input("Country (country)", value="id")

    st.markdown("---")
    run_scrape = st.button("🕷️ Ambil Reviews")

    if run_scrape:
        if not package_name.strip():
            st.error("Package Name tidak boleh kosong.")
        else:
            try:
                with st.spinner("Mengambil review dari Google Play..."):
                    sort_val = Sort.NEWEST if sort_choice == "NEWEST" else Sort.MOST_RELEVANT

                    rv, _ = reviews(
                        package_name,
                        lang=lang,
                        country=country,
                        sort=sort_val,
                        count=int(count)
                    )

                df_reviews = pd.DataFrame(rv)

                if df_reviews.empty:
                    st.warning("Tidak ada review yang berhasil diambil. Cek package name atau coba sort lain.")
                    st.session_state.scraped_df = None
                    st.session_state.scrape_meta = None
                else:
                    st.success(f"Berhasil mengambil {len(df_reviews)} review.")
                    st.session_state.scraped_df = df_reviews

                    st.session_state.scrape_meta = {
                        "package_name": package_name,
                        "count": int(count),
                        "sort": sort_choice,
                        "lang": lang,
                        "country": country,
                        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

            except Exception as e:
                st.error(f"Error fetching reviews: {e}")
                st.session_state.scraped_df = None
                st.session_state.scrape_meta = None

    # ===== tampilkan hasil scraping kalau ada =====
    if st.session_state.scraped_df is not None:
        df_reviews = st.session_state.scraped_df.copy()

        st.markdown("---")
        st.subheader("Preview Hasil Scraping")
        st.dataframe(df_reviews.head(30), use_container_width=True)

        # pilih kolom teks yang akan dijadikan "content"
        # google_play_scraper biasanya punya kolom 'content'
        text_cols = df_reviews.columns.tolist()
        default_idx = text_cols.index("content") if "content" in text_cols else 0

        chosen_text_col = st.selectbox(
            "Pilih kolom teks ulasan untuk dipakai sebagai dataset",
            options=text_cols,
            index=default_idx
        )

        # tombol: pakai langsung sebagai dataset
        if st.button("✅ Gunakan sebagai dataset → lanjut Proses"):
            work = pd.DataFrame({"content": df_reviews[chosen_text_col].astype(str)})
            work = drop_empty_rows(work)

            # set dataset kerja seperti menu Input
            st.session_state.raw_df = df_reviews.copy()
            st.session_state.df_work = work
            st.session_state.chosen_col = chosen_text_col

            # reset semua hasil preprocessing + model (biar konsisten)
            for k in ["pp_casefold","pp_normal","pp_clean","pp_stop","pp_stem","pp_filterlex","pp_labeled",
                      "tfidf","tfidf_df","X_tfidf","X_train","X_test","y_train","y_test",
                      "svm","y_pred","report","cm"]:
                st.session_state[k] = None

            st.success("Dataset dari scraping sudah dipakai. Mengalihkan ke menu Proses...")
            st.session_state.menu = "Proses"
            st.rerun()

        # opsi download tetap ada (opsional)
        st.markdown("### Download Dataset (Opsional)")
        meta = st.session_state.scrape_meta or {}
        fname = f"{meta.get('package_name','reviews')}_reviews_{meta.get('count',len(df_reviews))}.csv"
        csv_bytes = df_reviews.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Download CSV hasil scraping",
            data=csv_bytes,
            file_name=fname,
            mime="text/csv"
        )



# =========================================================
# INPUT
# =========================================================
elif st.session_state.menu == "Input":
    st.title("Input Data")
    st.write("Unggah file **CSV** dan pilih kolom yang berisi teks ulasan (nama kolom bebas).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=",", skipinitialspace=True, na_values="?")
        st.session_state.raw_df = df.copy()

        show_preview(df, "Preview data (raw)", n=20)

        chosen = st.selectbox("Pilih kolom teks ulasan", options=df.columns.tolist())
        if st.button("✅ Gunakan kolom ini"):
            work = pd.DataFrame({"content": df[chosen].astype(str)})
            work = drop_empty_rows(work)

            # reset downstream states
            for k in ["pp_casefold","pp_normal","pp_clean","pp_stop","pp_stem","pp_filterlex","pp_labeled",
                      "tfidf","tfidf_df","X_tfidf","X_train","X_test","y_train","y_test",
                      "svm","y_pred","report","cm"]:
                st.session_state[k] = None

            st.session_state.df_work = work
            st.session_state.chosen_col = chosen
            st.success("Data siap. Lanjut ke menu Proses.")
            st.session_state.menu = "Proses"
            st.rerun()

    if st.session_state.df_work is not None:
        show_preview(st.session_state.df_work, "Data yang akan diproses", n=20)


# =========================================================
# PROSES (Preprocessing)
# =========================================================
elif st.session_state.menu == "Proses":
    st.title("Proses (Preprocessing)")
    st.write("Di sini kamu bisa memilih menjalankan preprocessing **otomatis** atau **tahap per tahap**.")
   

    
    if st.session_state.df_work is None:
        st.warning("Belum ada data. Silakan ke menu Input.")
    elif st.session_state.res_errors:
        st.error("Resource gagal dimuat. Pastikan file kamus/lexicon ada di repo dan namanya sesuai.")
    else:
        base_df = st.session_state.df_work.copy()

        # steps
        def step_casefold(df):
            out = df.copy(); out["content"] = out["content"].apply(CaseFolding); return drop_empty_rows(out)
        def step_normalisasi(df):
            out = df.copy(); out["content"] = out["content"].apply(lambda x: normalisasi_dengan_kamus(x, st.session_state.kamus)); return drop_empty_rows(out)
        def step_clean(df):
            out = df.copy(); out["content"] = out["content"].apply(datacleaning); return drop_empty_rows(out)
        def step_stopword(df):
            out = df.copy(); out["content"] = out["content"].apply(remove_stopwords); return drop_empty_rows(out)
        def step_stemming(df):
            out = df.copy(); out["content"] = out["content"].apply(stem_text); return drop_empty_rows(out)
        def step_tokenizing(df):
            out = df.copy()
        
            def _ensure_list(x):
                if isinstance(x, list):
                    return x
                if pd.isna(x):
                    return []
                s = str(x).strip()
                if s.startswith("[") and s.endswith("]"):
                    s2 = s[1:-1].strip()
                    if not s2:
                        return []
                    parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
                    return [p for p in parts if p]
                return s.split()
        
            # kalau content_list sudah ada, amankan; kalau belum, buat dari content
            if "content_list" in out.columns:
                out["content_list"] = out["content_list"].apply(_ensure_list)
            else:
                out["content_list"] = out["content"].fillna("").astype(str).str.split()
        
            # Pastikan lower + strip + buang token kosong
            out["content_list"] = out["content_list"].apply(
                lambda toks: [str(t).strip().lower() for t in (toks or []) if str(t).strip()]
            )
        
            return out
        def step_filterlex(df):
            out = df.copy()
        
            if "content_list" not in out.columns:
                # pastikan tokenizing dulu
                out = step_tokenizing(out)
        
            lex_single = st.session_state.get("lexicon_single", None)
            if lex_single is None:
                # fallback: gabung pos+neg sederhana
                lex_single = {**st.session_state.lex_pos, **st.session_state.lex_neg}
        
            def filter_words_by_lexicon(word_list):
                if not isinstance(word_list, list):
                    return []
                return [w for w in word_list if w in lex_single]
        
            out["content_list"] = out["content_list"].apply(filter_words_by_lexicon)
        
            # gabungkan kembali ke content string
            out["content"] = out["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
        
            # drop NaN & list kosong (sesuai script)
            out = out.dropna(subset=["content"])
            out = out[out["content_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)
        
            return out

            return drop_empty_rows(out)
        def step_labeling(df):
            out = df.copy()
            if "content_list" not in out.columns:
                out = step_tokenizing(out)
        
            results = out["content_list"].apply(sentiment_analysis_lexicon_indonesia)
            out["score"] = results.apply(lambda x: x[0])
            out["Sentimen"] = results.apply(lambda x: x[1])
        
            return out



        # pilihan mode di MENU PROSES (bukan sidebar)
        st.markdown("### Pilih cara menjalankan preprocessing")
        st.markdown("<div class='seg-sub'>Pilih mode kerja: otomatis sekali jalan atau tahap per tahap.</div>", unsafe_allow_html=True)
        
        # pastikan ada default
        if "mode_proses" not in st.session_state or st.session_state.mode_proses is None:
            st.session_state.mode_proses = "auto"
        
        c1, c2 = st.columns(2)
        
        with c1:
            active_class = "seg-btn seg-active" if st.session_state.mode_proses == "auto" else "seg-btn"
            st.markdown(f"<div class='{active_class}'>", unsafe_allow_html=True)
            if st.button("⚡ Otomatis (run all)", use_container_width=True, key="btn_mode_auto"):
                st.session_state.mode_proses = "auto"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with c2:
            active_class = "seg-btn seg-active" if st.session_state.mode_proses == "step" else "seg-btn"
            st.markdown(f"<div class='{active_class}'>", unsafe_allow_html=True)
            if st.button("🧩 Tahap per tahap", use_container_width=True, key="btn_mode_step"):
                st.session_state.mode_proses = "step"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")


         #Reset Proses
        st.markdown("### Reset Proses")
        colR1, colR2 = st.columns([1, 3])
        
        with colR1:
            if st.button("🧹 Reset preprocessing"):
                # reset semua output preprocessing
                for k in ["pp_casefold","pp_normal","pp_clean","pp_stop","pp_stem","pp_token","pp_filterlex","pp_labeled","pp_labeled_raw"]:
                    st.session_state[k] = None
        
                # reset juga downstream model biar konsisten
                for k in ["tfidf","tfidf_df","X_tfidf","X_train","X_test","y_train","y_test",
                          "svm","y_pred","report","cm"]:
                    st.session_state[k] = None
        
                st.success("Preprocessing (dan hasil model) berhasil di-reset. Kamu bisa mulai ulang.")
                st.rerun()
        
        with colR2:
            st.markdown(
                "<div class='hint'>Gunakan reset ini jika salah urutan tahap atau ingin mengulang dari awal tanpa upload ulang.</div>",
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Tombol Otomatis
        if st.session_state.mode_proses == "auto":
            if st.button("▶️ Jalankan preprocessing (otomatis)"):
                progress = st.progress(0)
                with st.spinner("Menjalankan preprocessing..."):
                    st.session_state.pp_casefold = step_casefold(base_df);   progress.progress(15); time.sleep(0.05)
                    st.session_state.pp_normal = step_normalisasi(st.session_state.pp_casefold); progress.progress(30); time.sleep(0.05)
                    st.session_state.pp_clean = step_clean(st.session_state.pp_normal); progress.progress(45); time.sleep(0.05)
                    st.session_state.pp_stop = step_stopword(st.session_state.pp_clean); progress.progress(60); time.sleep(0.05)
                    st.session_state.pp_stem = step_stemming(st.session_state.pp_stop); progress.progress(75); time.sleep(0.05)
                    st.session_state.pp_filterlex = step_filterlex(st.session_state.pp_stem); progress.progress(90); time.sleep(0.05)
                    st.session_state.pp_labeled = step_labeling(st.session_state.pp_filterlex); progress.progress(100)
        
                # ✅ simpan raw (supaya metrik Labeling tidak berubah saat filter netral)
                st.session_state.pp_labeled_raw = st.session_state.pp_labeled.copy()
        
                st.success("Preprocessing + labeling selesai.")
                st.rerun()  # ✅ optional tapi bagus supaya UI langsung render state terbaru
        
            # ✅ PREVIEW DIPINDAHKAN KE LUAR BLOK TOMBOL (biar tetap muncul setelah rerun / filter netral)
            if st.session_state.pp_labeled is not None:
                st.markdown("---")
                st.subheader("Preview Hasil Preprocessing (Otomatis)")
        
                df_prev = st.session_state.pp_labeled
                
                cols_show = [c for c in ["content", "content_list", "score", "Sentimen"] if c in df_prev.columns]
                st.dataframe(df_prev[cols_show].head(30), use_container_width=True)
                
        # Tahap per tahap
        else:
            # helper aman untuk pilih prev dataframe
            def pick_prev(*candidates):
                for c in candidates:
                    if c is not None:
                        return c
                return None
            
            # Tahap per tahap
            st.subheader("Tahapan (Tahap per tahap)")
            
            # 1) Case Folding
            with st.expander("1) Case Folding", expanded=False):
                if st.button("Jalankan Case Folding"):
                    with st.spinner("Case folding..."):
                        st.session_state.pp_casefold = step_casefold(base_df)
                if st.session_state.pp_casefold is not None:
                    show_preview(st.session_state.pp_casefold, "Hasil Case Folding", n=20)
                    # ✅ KETERANGAN JUMLAH DATA
                    show_processed_count(base_df, st.session_state.pp_casefold, title="Keterangan Jumlah Data - Case Folding")
            
            # 2) Normalisasi (butuh Case Folding)
            with st.expander("2) Normalisasi Kamus", expanded=False):
                btn_norm = st.button("Jalankan Normalisasi", disabled=st.session_state.pp_casefold is None)
                if st.session_state.pp_casefold is None:
                    st.info("Jalankan **Case Folding** dulu.")
                if btn_norm:
                    prev = pick_prev(st.session_state.pp_casefold, base_df)
                    with st.spinner("Normalisasi..."):
                        st.session_state.pp_normal = step_normalisasi(prev)
                if st.session_state.pp_normal is not None:
                    show_preview(st.session_state.pp_normal, "Hasil Normalisasi", n=20)
                    # ✅ KETERANGAN JUMLAH DATA
                    show_processed_count(st.session_state.pp_casefold, st.session_state.pp_normal, title="Keterangan Jumlah Data - Normalisasi")
            
            # 3) Cleaning (butuh Normalisasi)
            with st.expander("3) Data Cleansing", expanded=False):
                btn_clean = st.button("Jalankan Cleansing", disabled=st.session_state.pp_normal is None)
                if st.session_state.pp_normal is None:
                    st.info("Jalankan **Normalisasi** dulu.")
                if btn_clean:
                    prev = pick_prev(st.session_state.pp_normal, base_df)
                    with st.spinner("Cleansing..."):
                        st.session_state.pp_clean = step_clean(prev)
                if st.session_state.pp_clean is not None:
                    show_preview(st.session_state.pp_clean, "Hasil Data Cleansing", n=20)
                    # ✅ KETERANGAN JUMLAH DATA
                    show_processed_count(st.session_state.pp_normal, st.session_state.pp_clean, title="Keterangan Jumlah Data - Data Cleansing")
            
            # 4) Stopword (butuh Cleansing)
            with st.expander("4) Stopword Removal", expanded=False):
                btn_stop = st.button("Jalankan Stopword", disabled=st.session_state.pp_clean is None)
                if st.session_state.pp_clean is None:
                    st.info("Jalankan **Cleansing** dulu.")
                if btn_stop:
                    prev = pick_prev(st.session_state.pp_clean, base_df)
                    with st.spinner("Stopword removal..."):
                        st.session_state.pp_stop = step_stopword(prev)
                if st.session_state.pp_stop is not None:
                    show_preview(st.session_state.pp_stop, "Hasil Stopword Removal", n=20)
                    # ✅ KETERANGAN JUMLAH DATA
                    show_processed_count(st.session_state.pp_clean, st.session_state.pp_stop, title="Keterangan Jumlah Data - Stopword Removal")
            
            # 5) Stemming (butuh Stopword)
            with st.expander("5) Stemming", expanded=False):
                btn_stem = st.button("Jalankan Stemming", disabled=st.session_state.pp_stop is None)
                if st.session_state.pp_stop is None:
                    st.info("Jalankan **Stopword Removal** dulu.")
                if btn_stem:
                    prev = pick_prev(st.session_state.pp_stop, base_df)
                    with st.spinner("Stemming..."):
                        st.session_state.pp_stem = step_stemming(prev)
                if st.session_state.pp_stem is not None:
                    show_preview(st.session_state.pp_stem, "Hasil Stemming", n=20)
                    # ✅ KETERANGAN JUMLAH DATA
                    show_processed_count(st.session_state.pp_stop, st.session_state.pp_stem, title="Keterangan Jumlah Data - Stemming")

            # 6) Tokenizing (butuh Stemming)
            with st.expander("6) Tokenizing", expanded=False):
                btn_tok = st.button("Jalankan Tokenizing", disabled=st.session_state.pp_stem is None)
                if st.session_state.pp_stem is None:
                    st.info("Jalankan **Stemming** dulu.")
                if btn_tok:
                    prev = pick_prev(st.session_state.pp_stem, base_df)
                    with st.spinner("Tokenizing..."):
                        st.session_state.pp_token = step_tokenizing(prev)
                        
            
                if st.session_state.pp_token is not None:
                    show_preview(st.session_state.pp_token, "Hasil Tokenizing", n=20)
                    # ✅ KETERANGAN JUMLAH DATA
                    show_processed_count(st.session_state.pp_stem, st.session_state.pp_token, title="Keterangan Jumlah Data - Tokenizing")
                    # tampilkan contoh token 5 baris pertama biar jelas tokennya
                    st.markdown("**Contoh token (5 baris pertama):**")
                    st.write(st.session_state.pp_token["content_list"].head(5))
            
            # 7) Filter Lexicon (butuh Tokenizing)
            with st.expander("7) Filter Lexicon (hapus typo/OOV)", expanded=False):
                btn_flex = st.button("Jalankan Filter Lexicon", disabled=st.session_state.pp_token is None)
                if st.session_state.pp_token is None:
                    st.info("Jalankan **Tokenizing** dulu.")
                if btn_flex:
                    prev = pick_prev(st.session_state.pp_token, base_df)
                    with st.spinner("Filter lexicon..."):
                        st.session_state.pp_filterlex = step_filterlex(prev)
                if st.session_state.pp_filterlex is not None:
                    show_preview(st.session_state.pp_filterlex, "Hasil Filter Lexicon", n=20)
                    # ✅ KETERANGAN JUMLAH DATA
                    show_processed_count(st.session_state.pp_token, st.session_state.pp_filterlex, title="Keterangan Jumlah Data - Filter Lexicon")
                        
            # 8) Labeling (butuh Filter Lexicon)
            with st.expander("8) Labeling Lexicon", expanded=False):
                btn_lab = st.button("Jalankan Labeling", disabled=st.session_state.pp_filterlex is None)
                if st.session_state.pp_filterlex is None:
                    st.info("Jalankan **Filter Lexicon** dulu.")
                if btn_lab:
                    prev = pick_prev(st.session_state.pp_filterlex, base_df)
                    with st.spinner("Labeling..."):
                        st.session_state.pp_labeled = step_labeling(prev)
                        st.session_state.pp_labeled_raw = st.session_state.pp_labeled.copy()  # ✅ simpan versi asli
                if st.session_state.pp_labeled is not None:
                    show_preview(st.session_state.pp_labeled, "Hasil Labeling", n=20)
                     # ✅ KETERANGAN JUMLAH DATA
                    show_processed_count(
                        st.session_state.pp_filterlex,
                        st.session_state.pp_labeled_raw if st.session_state.pp_labeled_raw is not None else st.session_state.pp_labeled,
                        title="Keterangan Jumlah Data - Labeling"
                    )



        # Summary + chart + download
        if st.session_state.pp_labeled is not None:
            st.markdown("---")
            st.subheader("Ringkasan Hasil Labeling (Lexicon)")

            # selalu ambil versi TERBARU dari session_state
            df_lab = st.session_state.pp_labeled.copy()
            show_processed_count(base_df, df_lab, title="Keterangan Jumlah Data - Total Setelah Preprocessing")
            
            # tombol filter netral
            if st.button("Filter netral (score == 0)"):
                df_current = st.session_state.pp_labeled.copy()
                before_n = len(df_current)
            
                after_df = df_current[df_current["score"] != 0].reset_index(drop=True)
                removed = before_n - len(after_df)
            
                # ✅ hanya ubah versi working
                st.session_state.pp_labeled = after_df
            
                st.success(f"Netral dihapus (untuk proses SVM). Terhapus: {removed} data.")
                st.rerun()
            
            # setelah mungkin rerun, ambil lagi data terbaru untuk render
            df_lab = st.session_state.pp_labeled
            
            col1, col2 = st.columns([1, 1])
            with col1:
                # ✅ ini otomatis hilangkan baris 'netral' kalau sudah terhapus
                st.write(df_lab["Sentimen"].value_counts().rename_axis("Sentimen").reset_index(name="count"))
            
            with col2:
                # ✅ plot otomatis menyesuaikan data yang tersisa
                plot_bar_counts(df_lab["Sentimen"], "Distribusi Sentimen (Lexicon)")

            excel_bytes = to_excel_bytes(st.session_state.pp_labeled, sheet_name="preprocessing")
            st.download_button(
                "⬇️ Download hasil preprocessing (Excel)",
                data=excel_bytes,
                file_name="hasil_preprocessing.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            if st.button("➡️ Lanjut ke Klasifikasi SVM"):
                st.session_state.menu = "Klasifikasi SVM"
                st.rerun()
        else:
            st.info("Jalankan preprocessing sampai labeling untuk melihat ringkasan & download.")


# =========================================================
# KLASIFIKASI SVM (Parameter pindah ke sini, CV dihapus)
# =========================================================
elif st.session_state.menu == "Klasifikasi SVM":
    st.title("Klasifikasi SVM")
    st.write("Tahapan: **TF-IDF → Split data → SVM → Evaluasi**.")

    if st.session_state.pp_labeled is None:
        st.warning("Data belum preprocessing+labeling. Silakan ke menu Proses.")
    else:
        df = st.session_state.pp_labeled.copy()
        if "Sentimen" not in df.columns:
            st.error("Kolom 'Sentimen' tidak ditemukan. Jalankan labeling di menu Proses.")
        else:
            df = df[df["Sentimen"].isin(["negatif", "positif"])].reset_index(drop=True)
            if df.empty:
                st.error("Data negatif/positif kosong. Pastikan setelah filter netral masih ada data.")
            else:
                if "content_list" not in df.columns:
                    df["content_list"] = df["content"].astype(str).str.split()

                # PARAMETER MODEL di sini
                st.markdown("### Parameter Model (di menu Klasifikasi SVM)")
                with st.expander("Atur parameter", expanded=True):
                    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
                    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
                    C = st.number_input("C", min_value=0.01, value=1.0, step=0.1)
                    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], index=0)

                st.markdown("---")

                # ===== GUARD / PRASYARAT =====
                can_tfidf = True  # di tahap ini df sudah ada (neg/pos)
                can_split = st.session_state.X_tfidf is not None
                can_svm = (st.session_state.X_train is not None) and (st.session_state.X_test is not None)

                colA, colB, colC, colD = st.columns([1.2, 1.2, 1.2, 2])

                with colA:
                    do_tfidf = st.button("1) TF-IDF", disabled=not can_tfidf)

                with colB:
                    do_split = st.button("2) Split", disabled=not can_split)

                with colC:
                    do_svm = st.button("3) SVM", disabled=not can_svm)

                with colD:
                    # hint status prasyarat (UX)
                    if not can_split:
                        st.markdown("<div class='hint'>Split terkunci: jalankan <b>TF-IDF</b> dulu.</div>", unsafe_allow_html=True)
                    elif not can_svm:
                        st.markdown("<div class='hint'>SVM terkunci: jalankan <b>Split</b> dulu.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='hint'>Semua tahapan siap dijalankan.</div>", unsafe_allow_html=True)

                # TF-IDF
                if do_tfidf:
                    progress = st.progress(0)
                    with st.spinner("Menghitung TF-IDF..."):
                        X_text = df["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
                        progress.progress(30)

                        tfidf = TfidfVectorizer()
                        X_tfidf = tfidf.fit_transform(X_text).toarray()

                        progress.progress(90)
                        tfidf_df = pd.DataFrame(X_tfidf, columns=tfidf.get_feature_names_out())
                        progress.progress(100)

                    st.session_state.tfidf = tfidf
                    st.session_state.X_tfidf = X_tfidf
                    st.session_state.tfidf_df = tfidf_df

                    # reset downstream (karena fitur berubah)
                    for k in ["X_train", "X_test", "y_train", "y_test", "svm", "y_pred", "report", "cm"]:
                        st.session_state[k] = None

                    st.success(f"TF-IDF selesai. Jumlah fitur: {tfidf_df.shape[1]}")
                    st.rerun()  # supaya tombol Split langsung aktif

                # TF-IDF preview
                if st.session_state.tfidf_df is not None:
                    st.markdown("---")
                    st.subheader("Hasil TF-IDF (Preview)")
                    c1, c2, c3 = st.columns([1.2, 1.2, 2])
                    with c1:
                        preview_rows = st.slider("Baris (preview)", 5, 100, 20, 5)
                    with c2:
                        preview_cols = st.slider("Fitur/kolom (preview)", 10, 300, 50, 10)
                    with c3:
                        st.markdown("<div class='hint'>Tabel preview agar aplikasi ringan. File lengkap dapat diunduh.</div>", unsafe_allow_html=True)

                    tfidf_df = st.session_state.tfidf_df
                    preview_df = tfidf_df.iloc[:preview_rows, :min(preview_cols, tfidf_df.shape[1])]
                    st.dataframe(preview_df, use_container_width=True)

                    tfidf_excel = to_excel_bytes(tfidf_df, sheet_name="tfidf")
                    st.download_button(
                        "⬇️ Download TF-IDF lengkap (Excel)",
                        data=tfidf_excel,
                        file_name="hasil_tfidf.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # Split
                if do_split:
                    with st.spinner("Melakukan split data..."):
                        y = df["Sentimen"]
                        X_train, X_test, y_train, y_test = train_test_split(
                            st.session_state.X_tfidf, y,
                            test_size=float(test_size),
                            random_state=int(random_state)
                        )
                    st.session_state.X_train, st.session_state.X_test = X_train, X_test
                    st.session_state.y_train, st.session_state.y_test = y_train, y_test
                    st.success(f"Split selesai. Train: {len(y_train)} | Test: {len(y_test)}")
                    st.rerun()  # supaya tombol SVM langsung aktif

                # SVM
                if do_svm:
                    progress = st.progress(0)
                    with st.spinner("Melatih model SVM..."):
                        progress.progress(20)
                        svm = SVC(kernel=kernel, C=float(C))
                        svm.fit(st.session_state.X_train, st.session_state.y_train)

                        progress.progress(70)
                        y_pred = svm.predict(st.session_state.X_test)

                        progress.progress(100)

                    st.session_state.svm = svm
                    st.session_state.y_pred = y_pred

                    acc = accuracy_score(st.session_state.y_test, y_pred)
                    st.session_state.report = classification_report(st.session_state.y_test, y_pred, zero_division=0)
                    st.session_state.cm = confusion_matrix(st.session_state.y_test, y_pred, labels=["negatif", "positif"])
                    st.success(f"SVM selesai. Accuracy: {acc:.4f}")

                # Results
                if st.session_state.report is not None and st.session_state.cm is not None:
                    st.markdown("---")
                    st.subheader("Ringkasan Hasil (Summary Cards)")
                    
                    acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)

                    y_pred_series = pd.Series(st.session_state.y_pred)
                    majority = y_pred_series.value_counts().idxmax()
                    maj_pct = y_pred_series.value_counts(normalize=True).max() * 100

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accuracy (Test)", f"{acc:.4f}")
                    m2.metric("Mayoritas Prediksi (Test)", f"{majority}")
                    m3.metric("Proporsi Mayoritas", f"{maj_pct:.1f}%")

                    st.subheader("Classification Report")
                    st.code(st.session_state.report)

                    st.subheader("Confusion Matrix")
                    plot_confusion(st.session_state.cm, labels=("negatif", "positif"), title="Confusion Matrix SVM")
                    st.info(biggest_confusion_insight(st.session_state.cm, labels=("negatif", "positif")))

                    st.subheader("Distribusi Prediksi (Data Uji)")
                    plot_bar_counts(pd.Series(st.session_state.y_pred), "Distribusi Prediksi (Test)")

                    # =========================================================
                    # KESIMPULAN OTOMATIS BERDASARKAN HASIL SVM
                    # =========================================================
                    st.markdown("---")
                    st.subheader("Kesimpulan Hasil Klasifikasi Sentimen")
                    
                    # ambil data uji & prediksi
                    y_test = st.session_state.y_test
                    y_pred = pd.Series(st.session_state.y_pred)
                    
                    total_test = len(y_test)
                    dist = y_pred.value_counts()
                    dist_pct = y_pred.value_counts(normalize=True) * 100
                    
                    acc = accuracy_score(y_test, st.session_state.y_pred)
                    
                    # tentukan sentimen dominan
                    dominant_sentiment = dist.idxmax()
                    dominant_pct = dist_pct.max()
                    
                    # narasi kesimpulan
                    if dominant_sentiment == "positif":
                        trend_sentence = "cenderung memberikan sentimen **positif** terhadap aplikasi yang dianalisis."
                    else:
                        trend_sentence = "cenderung memberikan sentimen **negatif** terhadap aplikasi yang dianalisis."
                    
                    st.markdown(
                        f"""
                    Berdasarkan hasil klasifikasi menggunakan metode **Support Vector Machine (SVM)** terhadap 
                    **{total_test} data uji**, diperoleh bahwa sentimen pengguna yang paling dominan adalah 
                    **{dominant_sentiment}** dengan proporsi sebesar **{dominant_pct:.1f}%** dari total data uji.
                    
                    Model SVM yang digunakan pada penelitian ini menghasilkan nilai **akurasi sebesar {acc:.4f}**, 
                    yang menunjukkan bahwa model memiliki kemampuan yang **cukup baik** dalam mengklasifikasikan 
                    sentimen ulasan pengguna ke dalam kelas positif dan negatif.
                    
                    Secara umum, hasil ini menunjukkan bahwa pengguna {trend_sentence}
                    """
                    )
                    
                    # catatan tambahan jika akurasi rendah
                    if acc < 0.6:
                        st.warning(
                            "Catatan: Nilai akurasi model relatif rendah. "
                            "Disarankan untuk menambah jumlah data latih, "
                            "melakukan tuning parameter, atau memperkaya proses preprocessing."
                        )
                    elif acc >= 0.8:
                        st.success(
                            "Model menunjukkan performa yang sangat baik dan hasil klasifikasi dapat dipercaya."
                        )


                # Downloads + save model
                st.markdown("---")
                st.subheader("Unduh Hasil & Model")

                if st.session_state.svm is None or st.session_state.tfidf is None:
                    st.info("Jalankan minimal TF-IDF → Split → SVM untuk mengunduh hasil dan model.")
                else:
                    X_all = st.session_state.tfidf.transform(
                        df["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
                    ).toarray()
                    df_out = df.copy()
                    df_out["Prediksi_SVM"] = st.session_state.svm.predict(X_all)

                    excel_bytes = to_excel_bytes(df_out, sheet_name="svm_results")
                    st.download_button(
                        "⬇️ Download hasil klasifikasi (Excel)",
                        data=excel_bytes,
                        file_name="hasil_klasifikasi_svm.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    bundle = make_model_bundle(st.session_state.tfidf, st.session_state.svm)
                    pkl_bytes = pickle.dumps(bundle)
                    st.download_button(
                        "⬇️ Download model (TF-IDF + SVM) .pkl",
                        data=pkl_bytes,
                        file_name="model_tfidf_svm.pkl",
                        mime="application/octet-stream"
                    )
