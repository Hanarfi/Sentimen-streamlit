# app.py
# =========================================================
# STREAMLIT APP (Bright + Friendly + Beginner/Advanced Mode)
# =========================================================

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
# CONFIG
# =========================================================
st.set_page_config(page_title="Analisis Sentimen Ulasan (Mudah)", layout="wide")


# =========================================================
# UI STYLE (Bright, friendly)
# =========================================================
def inject_bright_friendly():
    st.markdown(
        """
        <style>
            .stApp {
                background: #F7F8FC;
                color: #111827;
            }

            section[data-testid="stSidebar"]{
                background: #FFFFFF;
                border-right: 1px solid rgba(17,24,39,0.08);
            }

            h1,h2,h3,h4 { color:#0F172A !important; }
            p, li, label, div, span { color: #111827; }

            .card {
                background: #FFFFFF;
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 16px;
                padding: 14px 16px;
                box-shadow: 0 6px 20px rgba(15,23,42,0.04);
            }
            .muted { color: rgba(17,24,39,0.65); }
            .hint  { color: rgba(17,24,39,0.60); font-size: 0.93rem; }

            .primary-btn button {
                width: 100%;
                background: #2563EB !important;
                color: #FFFFFF !important;
                border: 1px solid rgba(37,99,235,0.25) !important;
                border-radius: 14px !important;
                padding: 0.75rem 1rem !important;
                font-weight: 800 !important;
            }
            .primary-btn button:hover {
                filter: brightness(1.06);
            }

            .soft-btn button {
                width: 100%;
                background: #FFFFFF !important;
                color: #111827 !important;
                border: 1px solid rgba(15,23,42,0.12) !important;
                border-radius: 14px !important;
                padding: 0.65rem 1rem !important;
                font-weight: 750 !important;
            }
            .soft-btn button:hover {
                border-color: rgba(15,23,42,0.22) !important;
            }

            .kpi {
                background: #FFFFFF;
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 16px;
                padding: 12px 14px;
                box-shadow: 0 6px 20px rgba(15,23,42,0.04);
            }

            div[data-testid="stDataFrame"]{
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 14px;
                overflow: hidden;
                background: #FFFFFF;
            }

            hr { border: none; border-top: 1px solid rgba(15,23,42,0.10); margin: 1rem 0; }

            .step-pill {
                display:inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid rgba(15,23,42,0.10);
                background: #FFFFFF;
                font-weight: 700;
                margin-right: 8px;
            }
            .step-ok { background: #ECFDF5; border-color: rgba(16,185,129,0.20); color:#065F46; }
            .step-now { background: #EFF6FF; border-color: rgba(37,99,235,0.22); color:#1D4ED8; }
            .step-lock { background: #F8FAFC; color: rgba(15,23,42,0.45); }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_bright_friendly()


# =========================================================
# NLTK setup (cloud-friendly)
# =========================================================
@st.cache_resource
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


ensure_nltk()


# =========================================================
# RESOURCES (repo files)
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
    lex = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            word = row[0].strip().lower()
            weight = int(row[1].strip())
            lex[word] = weight
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
# PREPROCESSING
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
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]", "", text)
    text = text.replace("\n", " ").strip(" ")
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


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["content"] = out["content"].fillna("").astype(str)
    out = out[out["content"].str.strip() != ""].reset_index(drop=True)
    return out


def ensure_list(x):
    if isinstance(x, list):
        return x
    return str(x).split()


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
    plt.xlabel("Kategori")
    plt.ylabel("Jumlah")
    st.pyplot(fig)
    plt.close(fig)


def plot_confusion(cm, labels=("negatif", "positif"), title="Confusion Matrix"):
    cm = np.array(cm)
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(labels), yticklabels=list(labels))
    plt.xlabel("Prediksi")
    plt.ylabel("Asli")
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)


def biggest_confusion_insight(cm, labels=("negatif", "positif")) -> str:
    cm = np.array(cm)
    if cm.shape != (2, 2):
        return "Insight belum tersedia."
    fn = cm[0, 1]  # true negatif diprediksi positif
    fp = cm[1, 0]  # true positif diprediksi negatif
    if fn == 0 and fp == 0:
        return "Bagus! Tidak ada kesalahan pada data uji."
    if fn >= fp:
        return f"Kesalahan paling sering: **{labels[0]} → {labels[1]}** sebanyak **{fn}**."
    return f"Kesalahan paling sering: **{labels[1]} → {labels[0]}** sebanyak **{fp}**."


def make_model_bundle(tfidf: TfidfVectorizer, svm: SVC):
    return {"tfidf": tfidf, "svm": svm}


# =========================================================
# PIPELINE FUNCTIONS (friendly + robust)
# =========================================================
def run_preprocessing_pipeline(df_in: pd.DataFrame, kamus: dict, lex_pos: dict, lex_neg: dict) -> pd.DataFrame:
    df = df_in.copy()
    df = drop_empty_rows(df)

    # 1) casefold
    df["content"] = df["content"].apply(CaseFolding)

    # 2) normalize
    df["content"] = df["content"].apply(lambda x: normalisasi_dengan_kamus(x, kamus))

    # 3) clean
    df["content"] = df["content"].apply(datacleaning)

    # 4) stopwords
    df["content"] = df["content"].apply(remove_stopwords)

    # 5) stemming
    df["content"] = df["content"].apply(stem_text)

    # 6) tokenize
    df["content_list"] = df["content"].fillna("").astype(str).str.lower().str.split()

    # 7) filter tokens by lexicon
    def filter_tokens(tokens):
        tokens = ensure_list(tokens)
        tokens = [str(t).strip().lower() for t in (tokens or []) if str(t).strip()]
        return [w for w in tokens if (w in lex_pos) or (w in lex_neg)]

    df["content_list"] = df["content_list"].apply(filter_tokens)
    df = df[df["content_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)

    # 8) labeling
    def score_sent(tokens):
        score = 0
        for w in tokens:
            if w in lex_pos:
                score += lex_pos[w]
            if w in lex_neg:
                score += lex_neg[w]
        if score > 0:
            sent = "positif"
        elif score < 0:
            sent = "negatif"
        else:
            sent = "netral"
        return score, sent

    res = df["content_list"].apply(score_sent)
    df["score"] = res.apply(lambda x: x[0])
    df["Sentimen"] = res.apply(lambda x: x[1])
    df["content"] = df["content_list"].apply(lambda x: " ".join(x))

    return df


def run_training_pipeline(df_labeled: pd.DataFrame, test_size=0.2, random_state=42, C=1.0, kernel="linear"):
    # only pos/neg
    df = df_labeled[df_labeled["Sentimen"].isin(["negatif", "positif"])].reset_index(drop=True)
    if df.empty:
        raise ValueError("Data positif/negatif kosong. Pastikan ada cukup ulasan setelah labeling.")

    X_text = df["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    y = df["Sentimen"]

    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X_text).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=float(test_size), random_state=int(random_state)
    )

    svm = SVC(kernel=kernel, C=float(C))
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["negatif", "positif"])
    acc = accuracy_score(y_test, y_pred)

    return {
        "df_used": df,
        "tfidf": tfidf,
        "X_tfidf": X_tfidf,
        "svm": svm,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
        "cm": cm,
        "acc": acc,
    }


# =========================================================
# SESSION STATE
# =========================================================
def init_state():
    defaults = {
        # app settings
        "mode_user": "Pemula",  # Pemula / Lanjutan
        "step": "1) Ambil Data",

        # resources
        "kamus": None, "lex_pos": None, "lex_neg": None,
        "res_errors": [],

        # dataset
        "raw_df": None,
        "df_work": None,          # DataFrame with column "content"
        "chosen_col": None,

        # scraping
        "scraped_df": None,
        "scrape_meta": None,

        # results
        "df_labeled": None,
        "df_labeled_raw": None,   # before filtering netral (if any)
        "model_pack": None,       # dict from run_training_pipeline
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# Load resources once
if st.session_state.kamus is None or st.session_state.lex_pos is None or st.session_state.lex_neg is None:
    kamus, lex_pos, lex_neg, errors = safe_load_resources()
    st.session_state.kamus = kamus
    st.session_state.lex_pos = lex_pos
    st.session_state.lex_neg = lex_neg
    st.session_state.res_errors = errors


# =========================================================
# SIDEBAR (Wizard)
# =========================================================
st.sidebar.markdown("## Pengaturan")

st.session_state.mode_user = st.sidebar.radio(
    "Mode Pengguna",
    ["Pemula", "Lanjutan"],
    index=0 if st.session_state.mode_user == "Pemula" else 1
)

st.sidebar.markdown("---")
st.session_state.step = st.sidebar.radio(
    "Langkah",
    ["1) Ambil Data", "2) Bersihkan & Label", "3) Latih Model", "4) Hasil & Unduh"],
    index=["1) Ambil Data", "2) Bersihkan & Label", "3) Latih Model", "4) Hasil & Unduh"].index(st.session_state.step)
)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset Semua"):
    for k in list(st.session_state.keys()):
        if k in ["mode_user", "step"]:
            continue
        st.session_state[k] = None
    init_state()
    st.rerun()

# Resource status
st.sidebar.markdown("---")
st.sidebar.markdown("## Status Resource")
if st.session_state.res_errors:
    st.sidebar.error("Resource gagal dimuat.")
    for e in st.session_state.res_errors:
        st.sidebar.write(f"- {e}")
else:
    st.sidebar.success("Resource siap.")
    st.sidebar.write(f"- Kamus: {len(st.session_state.kamus)}")
    st.sidebar.write(f"- Lexicon +: {len(st.session_state.lex_pos)}")
    st.sidebar.write(f"- Lexicon -: {len(st.session_state.lex_neg)}")


# =========================================================
# HEADER + STEP INDICATOR
# =========================================================
def step_indicator():
    step_map = ["1) Ambil Data", "2) Bersihkan & Label", "3) Latih Model", "4) Hasil & Unduh"]
    current = st.session_state.step

    ok1 = st.session_state.df_work is not None
    ok2 = st.session_state.df_labeled is not None
    ok3 = st.session_state.model_pack is not None

    status = {
        "1) Ambil Data": ok1,
        "2) Bersihkan & Label": ok2,
        "3) Latih Model": ok3,
        "4) Hasil & Unduh": ok3,
    }

    pills = []
    for s in step_map:
        cls = "step-pill"
        if s == current:
            cls += " step-now"
        elif status.get(s, False):
            cls += " step-ok"
        else:
            cls += " step-lock"
        pills.append(f"<span class='{cls}'>{s}</span>")

    st.markdown("".join(pills), unsafe_allow_html=True)


st.title("Analisis Sentimen Ulasan (Versi Ramah Pengguna)")
st.markdown(
    "<div class='card'>"
    "<b>Apa yang aplikasi ini lakukan?</b><br>"
    "<span class='muted'>Aplikasi ini membaca ulasan pengguna, lalu menebak apakah sentimennya <b>positif</b> atau <b>negatif</b>. "
    "Kamu cukup ikuti langkah 1 sampai 4. Tidak perlu paham istilah teknis.</span>"
    "</div>",
    unsafe_allow_html=True
)
st.write("")
step_indicator()
st.write("")


# =========================================================
# STEP 1) AMBIL DATA
# =========================================================
if st.session_state.step == "1) Ambil Data":
    st.subheader("Langkah 1 — Ambil Data Ulasan")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            "<div class='card'>"
            "<b>Pilih sumber data</b><br>"
            "<span class='hint'>Kamu bisa ambil langsung dari Google Play (opsional), atau upload CSV milikmu.</span>"
            "</div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            "<div class='card'>"
            "<b>Tips</b><br>"
            "<span class='hint'>Kalau CSV kamu punya kolom bernama <code>content</code> (atau mirip), pilih kolom itu.</span>"
            "</div>",
            unsafe_allow_html=True
        )

    st.write("")
    tabs = st.tabs(["🕷️ Ambil dari Google Play (Opsional)", "📥 Upload CSV"])

    # ---- Scrape tab ----
    with tabs[0]:
        st.markdown("#### Ambil ulasan dari Google Play")
        st.markdown("<span class='hint'>Masukkan package name aplikasi. Contoh: <code>co.id.bankbsi.superapp</code></span>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            package_name = st.text_input("Package Name", value="co.id.bankbsi.superapp")
        with col2:
            sort_choice = st.selectbox("Urutkan", ["Terbaru", "Paling Relevan"], index=0)
        with col3:
            count = st.number_input("Jumlah ulasan", min_value=10, max_value=500, value=100, step=10)

        lang = st.text_input("Bahasa (lang)", value="id")
        country = st.text_input("Negara (country)", value="id")

        st.write("")
        st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
        run_scrape = st.button("🕷️ Ambil Ulasan Sekarang")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_scrape:
            if not package_name.strip():
                st.error("Package Name tidak boleh kosong.")
            else:
                try:
                    with st.spinner("Mengambil ulasan dari Google Play..."):
                        sort_val = Sort.NEWEST if sort_choice == "Terbaru" else Sort.MOST_RELEVANT
                        rv, _ = reviews(
                            package_name,
                            lang=lang,
                            country=country,
                            sort=sort_val,
                            count=int(count),
                        )
                    df_reviews = pd.DataFrame(rv)
                    if df_reviews.empty:
                        st.warning("Tidak ada ulasan yang berhasil diambil. Coba cek package name atau ganti urutan.")
                        st.session_state.scraped_df = None
                        st.session_state.scrape_meta = None
                    else:
                        st.success(f"Berhasil mengambil {len(df_reviews)} ulasan.")
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

        if st.session_state.scraped_df is not None:
            df_reviews = st.session_state.scraped_df.copy()
            st.markdown("---")
            st.markdown("#### Preview hasil")
            st.dataframe(df_reviews.head(20), use_container_width=True)

            text_cols = df_reviews.columns.tolist()
            default_idx = text_cols.index("content") if "content" in text_cols else 0
            chosen_text_col = st.selectbox("Kolom teks ulasan", options=text_cols, index=default_idx)

            cA, cB = st.columns([1.2, 1])
            with cA:
                st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
                use_as_dataset = st.button("✅ Pakai ini sebagai dataset (lanjut ke Langkah 2)")
                st.markdown("</div>", unsafe_allow_html=True)
            with cB:
                meta = st.session_state.scrape_meta or {}
                fname = f"{meta.get('package_name','reviews')}_reviews_{meta.get('count',len(df_reviews))}.csv"
                csv_bytes = df_reviews.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=fname, mime="text/csv")

            if use_as_dataset:
                work = pd.DataFrame({"content": df_reviews[chosen_text_col].astype(str)})
                work = drop_empty_rows(work)
                st.session_state.raw_df = df_reviews.copy()
                st.session_state.df_work = work
                st.session_state.chosen_col = chosen_text_col

                # reset downstream
                st.session_state.df_labeled = None
                st.session_state.df_labeled_raw = None
                st.session_state.model_pack = None

                st.session_state.step = "2) Bersihkan & Label"
                st.rerun()

    # ---- Upload tab ----
    with tabs[1]:
        st.markdown("#### Upload dataset CSV milikmu")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded, sep=",", skipinitialspace=True, na_values="?")
            st.session_state.raw_df = df.copy()

            st.markdown("##### Preview data (awal)")
            st.dataframe(df.head(20), use_container_width=True)

            # smart default for column
            cols = df.columns.tolist()
            preferred = None
            for cand in ["content", "review", "ulasan", "text", "komentar", "comments"]:
                if cand in cols:
                    preferred = cand
                    break
            chosen = st.selectbox(
                "Pilih kolom teks ulasan",
                options=cols,
                index=cols.index(preferred) if preferred in cols else 0,
            )

            st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
            use_col = st.button("✅ Gunakan kolom ini (lanjut ke Langkah 2)")
            st.markdown("</div>", unsafe_allow_html=True)

            if use_col:
                work = pd.DataFrame({"content": df[chosen].astype(str)})
                work = drop_empty_rows(work)

                st.session_state.df_work = work
                st.session_state.chosen_col = chosen

                st.session_state.df_labeled = None
                st.session_state.df_labeled_raw = None
                st.session_state.model_pack = None

                st.session_state.step = "2) Bersihkan & Label"
                st.rerun()

    if st.session_state.df_work is not None:
        st.markdown("---")
        st.success(f"Dataset siap ✅ (jumlah data: {len(st.session_state.df_work)})")


# =========================================================
# STEP 2) BERSIHKAN & LABEL
# =========================================================
elif st.session_state.step == "2) Bersihkan & Label":
    st.subheader("Langkah 2 — Bersihkan Teks & Beri Label Otomatis")

    if st.session_state.df_work is None:
        st.warning("Belum ada data. Kembali ke Langkah 1 untuk ambil/upload dataset.")
    elif st.session_state.res_errors:
        st.error("Resource gagal dimuat. Pastikan file kamus/lexicon ada di folder project.")
    else:
        st.markdown(
            "<div class='card'>"
            "<b>Apa yang terjadi di langkah ini?</b><br>"
            "<span class='hint'>Kami membersihkan teks (hapus simbol/emoji/URL, stopword, stemming), "
            "lalu memberi label sentimen otomatis berdasarkan kamus kata positif & negatif.</span>"
            "</div>",
            unsafe_allow_html=True
        )
        st.write("")

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
            run_pp = st.button("🧼 Jalankan Pembersihan & Label (Otomatis)")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='soft-btn'>", unsafe_allow_html=True)
            go_back = st.button("⬅️ Kembali ke Langkah 1")
            st.markdown("</div>", unsafe_allow_html=True)
            if go_back:
                st.session_state.step = "1) Ambil Data"
                st.rerun()

        if run_pp:
            with st.spinner("Memproses data..."):
                df_labeled = run_preprocessing_pipeline(
                    st.session_state.df_work,
                    st.session_state.kamus,
                    st.session_state.lex_pos,
                    st.session_state.lex_neg,
                )
                st.session_state.df_labeled_raw = df_labeled.copy()
                st.session_state.df_labeled = df_labeled.copy()
                st.session_state.model_pack = None
            st.success("Selesai! Data sudah diberi label sentimen.")
            st.rerun()

        if st.session_state.df_labeled is not None:
            df_lab = st.session_state.df_labeled.copy()

            st.markdown("---")
            st.markdown("#### Ringkasan label sentimen")
            vc = df_lab["Sentimen"].value_counts().rename_axis("Sentimen").reset_index(name="Jumlah")
            cA, cB = st.columns([1, 1])
            with cA:
                st.dataframe(vc, use_container_width=True)
            with cB:
                plot_bar_counts(df_lab["Sentimen"], "Distribusi Sentimen (Hasil Label Otomatis)")

            # Friendly action: remove netral (recommended for SVM)
            netral_count = int((df_lab["Sentimen"] == "netral").sum())
            st.markdown(
                f"<div class='card'><b>Rekomendasi</b><br>"
                f"<span class='hint'>Untuk model klasifikasi (positif vs negatif), biasanya data <b>netral</b> diabaikan. "
                f"Netral saat ini: <b>{netral_count}</b> baris.</span></div>",
                unsafe_allow_html=True
            )
            st.write("")

            cX, cY, cZ = st.columns([1.2, 1.2, 1])
            with cX:
                st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
                remove_netral = st.button("🧽 Hapus Netral (Disarankan)")
                st.markdown("</div>", unsafe_allow_html=True)

            with cY:
                st.markdown("<div class='soft-btn'>", unsafe_allow_html=True)
                restore = st.button("↩️ Kembalikan (pakai versi asli)")
                st.markdown("</div>", unsafe_allow_html=True)

            with cZ:
                excel_bytes = to_excel_bytes(df_lab, sheet_name="preprocessing")
                st.download_button(
                    "⬇️ Download (Excel)",
                    data=excel_bytes,
                    file_name="hasil_preprocessing.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            if remove_netral:
                before_n = len(st.session_state.df_labeled)
                st.session_state.df_labeled = st.session_state.df_labeled[st.session_state.df_labeled["score"] != 0].reset_index(drop=True)
                removed = before_n - len(st.session_state.df_labeled)
                st.success(f"Netral dihapus: {removed} baris.")
                st.rerun()

            if restore and st.session_state.df_labeled_raw is not None:
                st.session_state.df_labeled = st.session_state.df_labeled_raw.copy()
                st.success("Data dikembalikan ke versi asli (termasuk netral).")
                st.rerun()

            # Show preview (friendly)
            st.markdown("---")
            st.markdown("#### Contoh hasil (beberapa baris)")
            cols_show = [c for c in ["content", "score", "Sentimen"] if c in df_lab.columns]
            st.dataframe(df_lab[cols_show].head(20), use_container_width=True)

            # Advanced details hidden
            if st.session_state.mode_user == "Lanjutan":
                with st.expander("🔧 Detail teknis preprocessing (mode lanjutan)", expanded=False):
                    st.write("Preview token (5 baris pertama):")
                    if "content_list" in df_lab.columns:
                        st.write(df_lab["content_list"].head(5))
                    st.write("Preview lengkap beberapa kolom:")
                    show_cols = [c for c in ["content", "content_list", "score", "Sentimen"] if c in df_lab.columns]
                    st.dataframe(df_lab[show_cols].head(30), use_container_width=True)

            st.write("")
            st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
            go_next = st.button("➡️ Lanjut ke Langkah 3 (Latih Model)")
            st.markdown("</div>", unsafe_allow_html=True)
            if go_next:
                st.session_state.step = "3) Latih Model"
                st.rerun()


# =========================================================
# STEP 3) LATIH MODEL
# =========================================================
elif st.session_state.step == "3) Latih Model":
    st.subheader("Langkah 3 — Latih Model & Uji Akurasi")

    if st.session_state.df_labeled is None:
        st.warning("Belum ada data hasil langkah 2. Silakan jalankan Bersihkan & Label dulu.")
    else:
        st.markdown(
            "<div class='card'>"
            "<b>Apa yang terjadi di langkah ini?</b><br>"
            "<span class='hint'>Kami melatih model untuk membedakan ulasan <b>positif</b> dan <b>negatif</b>, "
            "lalu menguji akurasinya pada sebagian data.</span>"
            "</div>",
            unsafe_allow_html=True
        )
        st.write("")

        # Parameters: hidden for beginner, available for advanced
        if st.session_state.mode_user == "Pemula":
            test_size = 0.2
            random_state = 42
            C = 1.0
            kernel = "linear"
            st.info("Mode Pemula: pengaturan model menggunakan nilai default yang aman. (Bisa diubah di Mode Lanjutan).")
        else:
            with st.expander("🔧 Pengaturan model (mode lanjutan)", expanded=True):
                test_size = st.slider("Porsi data uji (test size)", 0.1, 0.4, 0.2, 0.05)
                random_state = st.number_input("Random state", min_value=0, value=42, step=1)
                C = st.number_input("C (kekuatan penalti)", min_value=0.01, value=1.0, step=0.1)
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], index=0)

        st.write("")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
            train_btn = st.button("🤖 Latih Model & Tampilkan Hasil")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='soft-btn'>", unsafe_allow_html=True)
            back_btn = st.button("⬅️ Kembali ke Langkah 2")
            st.markdown("</div>", unsafe_allow_html=True)
            if back_btn:
                st.session_state.step = "2) Bersihkan & Label"
                st.rerun()

        if train_btn:
            try:
                with st.spinner("Melatih model..."):
                    pack = run_training_pipeline(
                        st.session_state.df_labeled,
                        test_size=test_size,
                        random_state=random_state,
                        C=C,
                        kernel=kernel,
                    )
                st.session_state.model_pack = pack
                st.success(f"Selesai! Akurasi data uji: {pack['acc']:.4f}")
                st.rerun()
            except Exception as e:
                st.error(f"Gagal melatih model: {e}")

        if st.session_state.model_pack is not None:
            pack = st.session_state.model_pack
            acc = pack["acc"]

            st.markdown("---")
            st.subheader("Ringkasan yang mudah dipahami")

            y_pred = pd.Series(pack["y_pred"])
            dist = y_pred.value_counts()
            dist_pct = y_pred.value_counts(normalize=True) * 100
            dominant = dist.idxmax()
            dominant_pct = float(dist_pct.max())
            total_test = len(pack["y_test"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Akurasi (data uji)", f"{acc:.4f}")
            m2.metric("Sentimen dominan (prediksi)", f"{dominant}")
            m3.metric("Proporsi dominan", f"{dominant_pct:.1f}%")

            if dominant == "positif":
                trend_sentence = "cenderung memberikan sentimen **positif**."
            else:
                trend_sentence = "cenderung memberikan sentimen **negatif**."

            st.markdown(
                f"""
                Dari **{total_test}** data uji, model memprediksi sentimen dominan adalah **{dominant}**
                dengan proporsi **{dominant_pct:.1f}%**. Secara umum, ulasan pengguna {trend_sentence}
                """
            )

            if acc < 0.6:
                st.warning(
                    "Akurasi masih rendah. Coba tambah jumlah data, perbaiki preprocessing, "
                    "atau gunakan Mode Lanjutan untuk tuning parameter."
                )
            elif acc >= 0.8:
                st.success("Performa model sangat baik. Hasil klasifikasi lebih dapat dipercaya.")

            st.markdown("---")
            st.subheader("Visual ringkas")
            cA, cB = st.columns([1, 1])
            with cA:
                plot_confusion(pack["cm"], labels=("negatif", "positif"), title="Confusion Matrix")
                st.info(biggest_confusion_insight(pack["cm"], labels=("negatif", "positif")))
            with cB:
                plot_bar_counts(y_pred, "Distribusi Prediksi (Data Uji)")

            # Technical details hidden
            if st.session_state.mode_user == "Lanjutan":
                with st.expander("🔧 Detail teknis evaluasi (mode lanjutan)", expanded=False):
                    st.markdown("**Classification Report**")
                    st.code(pack["report"])
                    st.markdown("**Contoh fitur (TF-IDF)**")
                    st.write(f"Jumlah fitur: {len(pack['tfidf'].get_feature_names_out())}")

            st.write("")
            st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
            next_btn = st.button("➡️ Lanjut ke Langkah 4 (Hasil & Unduh)")
            st.markdown("</div>", unsafe_allow_html=True)
            if next_btn:
                st.session_state.step = "4) Hasil & Unduh"
                st.rerun()


# =========================================================
# STEP 4) HASIL & UNDUH
# =========================================================
elif st.session_state.step == "4) Hasil & Unduh":
    st.subheader("Langkah 4 — Unduh Hasil dan Model")

    if st.session_state.model_pack is None:
        st.warning("Model belum dilatih. Silakan ke Langkah 3.")
    else:
        pack = st.session_state.model_pack
        df_used = pack["df_used"].copy()
        tfidf = pack["tfidf"]
        svm = pack["svm"]

        st.markdown(
            "<div class='card'>"
            "<b>Apa yang bisa kamu unduh?</b><br>"
            "<span class='hint'>1) Hasil prediksi untuk seluruh data, 2) Model (TF-IDF + SVM) untuk dipakai lagi nanti.</span>"
            "</div>",
            unsafe_allow_html=True
        )
        st.write("")

        # Predict for all rows
        X_all = tfidf.transform(
            df_used["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
        ).toarray()
        df_out = df_used.copy()
        df_out["Prediksi_SVM"] = svm.predict(X_all)

        # Friendly preview
        st.markdown("#### Preview hasil prediksi")
        show_cols = [c for c in ["content", "Sentimen", "Prediksi_SVM", "score"] if c in df_out.columns]
        st.dataframe(df_out[show_cols].head(20), use_container_width=True)

        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        with c1:
            excel_bytes = to_excel_bytes(df_out, sheet_name="svm_results")
            st.download_button(
                "⬇️ Download hasil klasifikasi (Excel)",
                data=excel_bytes,
                file_name="hasil_klasifikasi_svm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with c2:
            bundle = make_model_bundle(tfidf, svm)
            pkl_bytes = pickle.dumps(bundle)
            st.download_button(
                "⬇️ Download model (TF-IDF + SVM) .pkl",
                data=pkl_bytes,
                file_name="model_tfidf_svm.pkl",
                mime="application/octet-stream",
            )

        with c3:
            st.markdown("<div class='soft-btn'>", unsafe_allow_html=True)
            back = st.button("⬅️ Kembali ke Langkah 3")
            st.markdown("</div>", unsafe_allow_html=True)
            if back:
                st.session_state.step = "3) Latih Model"
                st.rerun()

        # Optional: TF-IDF download for advanced only (not shown for beginner)
        if st.session_state.mode_user == "Lanjutan":
            with st.expander("🔧 Unduh TF-IDF (mode lanjutan)", expanded=False):
                tfidf_df = pd.DataFrame(pack["X_tfidf"], columns=tfidf.get_feature_names_out())
                tfidf_excel = to_excel_bytes(tfidf_df, sheet_name="tfidf")
                st.download_button(
                    "⬇️ Download TF-IDF lengkap (Excel)",
                    data=tfidf_excel,
                    file_name="hasil_tfidf.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.markdown("<span class='hint'>Catatan: file ini bisa besar (banyak kolom fitur).</span>", unsafe_allow_html=True)
