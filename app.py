import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
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


# =========================
# NLTK setup (Cloud-friendly)
# =========================
@st.cache_resource
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

ensure_nltk()


# =========================
# --- FUNGSI PREPROCESSING (dari kode kamu, direvisi agar cloud-stable) ---
# =========================
def CaseFolding(text: str) -> str:
    return str(text).lower()

def datacleaning(text: str) -> str:
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)          # mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text)          # hashtag
    text = re.sub(r"[^a-z\s]", " ", text)              # angka & simbol
    text = re.sub(r'RT[\s]', '', text)                 # RT
    text = re.sub(r'RT[?|$|.|@!&:_=)(><,]', '', text)  # simbol
    text = re.sub(r'http\S+', '', text)                # link
    text = re.sub(r'[0-9]', '', text)                  # angka
    text = text.replace('\n', ' ')
    text = text.strip(' ')
    text = re.sub('s://t.co/', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('"', '')
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    # Revisi: pakai split() agar tidak butuh NLTK punkt/punkt_tab
    stop_words = set(stopwords.words('indonesian'))
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


# =========================
# LOAD KAMUS & LEXICON (UPLOAD)
# =========================
def load_kamus_from_excel(uploaded_excel) -> dict:
    df = pd.read_excel(uploaded_excel)
    # kolom harus: non_standard, standard_word
    kamus_dict = dict(zip(df['non_standard'], df['standard_word']))
    return kamus_dict

def normalisasi_dengan_kamus(text: str, kamus_dict: dict) -> str:
    words = str(text).split()
    normalized = [kamus_dict.get(w, w) for w in words]
    return " ".join(normalized)

def load_lexicon_csv(uploaded_csv) -> dict:
    lex = {}
    content = uploaded_csv.getvalue().decode("utf-8", errors="ignore").splitlines()
    reader = csv.reader(content, delimiter=',')
    for row in reader:
        if len(row) >= 2:
            try:
                lex[row[0]] = int(row[1])
            except:
                pass
    return lex

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

def filter_tokens_by_lexicon(tokens, lex_pos: dict, lex_neg: dict):
    """
    Tahap tambahan: hanya menyisakan kata yang ada di lexicon positif/negatif.
    Berguna untuk membuang typo/OOV (out-of-vocabulary).
    """
    if tokens is None:
        return []
    allowed = set(lex_pos.keys()) | set(lex_neg.keys())
    return [t for t in tokens if t in allowed]


# =========================
# UTIL UI
# =========================
def show_df(df, title, max_rows=200):
    st.subheader(title)
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df.head(max_rows))

def plot_confusion(cm, labels=("negatif", "positif"), title="Confusion Matrix"):
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(labels), yticklabels=list(labels))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Sentiment Analysis Step-by-Step", layout="wide")
st.title("Sentiment Analysis Ulasan (Step-by-Step) — TF-IDF + SVM")
st.markdown("""
Aplikasi ini menjalankan proses bertahap:
1) Upload data (kolom bebas) → 2) Preprocessing per tahap → 3) Filter Lexicon (hapus typo/OOV) →  
4) Lexicon Labeling → 5) TF-IDF → 6) Train/Test SVM → 7) Evaluasi → 8) Export
""")


# Sidebar: file upload
st.sidebar.header("Upload File")
data_file = st.sidebar.file_uploader("Upload CSV (nama kolom bebas, pilih kolom teks di STEP 0)", type=["csv"])
kamus_file = st.sidebar.file_uploader("Upload Kamus Kata Baku (Excel: non_standard, standard_word)", type=["xlsx"])
lex_pos_file = st.sidebar.file_uploader("Upload Lexicon Positive (CSV: kata,skor)", type=["csv"])
lex_neg_file = st.sidebar.file_uploader("Upload Lexicon Negative (CSV: kata,skor)", type=["csv"])

st.sidebar.divider()
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)
C = st.sidebar.number_input("SVM C", min_value=0.01, value=1.0, step=0.1)
kernel = st.sidebar.selectbox("SVM kernel", ["linear", "rbf", "poly", "sigmoid"], index=0)

st.sidebar.divider()
st.sidebar.caption("Catatan: Normalisasi butuh file kamus. Filter Lexicon & Labeling butuh lexicon pos+neg.")


# Session state init
def init_state():
    defaults = {
        "raw_df": None,
        "df_step": None,
        "chosen_text_column": None,
        "kamus_dict": None,
        "lex_pos": None,
        "lex_neg": None,
        "tfidf": None,
        "svm": None,
        "X_train": None, "X_test": None, "y_train": None, "y_test": None,
        "y_pred": None,
        "report": None,
        "cm": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# =========================
# STEP 0: Load dataset & pilih kolom teks (GENERIC)
# =========================
st.header("STEP 0 — Load Dataset (Kolom Bebas)")

if data_file is None:
    st.info("Upload CSV dulu lewat sidebar.")
else:
    df = pd.read_csv(data_file, sep=",", skipinitialspace=True, na_values="?")
    st.session_state.raw_df = df.copy()

    show_df(df, "Preview Dataset (Raw)")

    text_column = st.selectbox(
        "Pilih kolom yang berisi teks ulasan",
        options=df.columns.tolist(),
        index=0 if st.session_state.chosen_text_column is None else df.columns.tolist().index(st.session_state.chosen_text_column)
        if st.session_state.chosen_text_column in df.columns else 0
    )

    if st.button("Gunakan kolom ini sebagai teks ulasan"):
        st.session_state.chosen_text_column = text_column
        st.session_state.df_step = pd.DataFrame()
        st.session_state.df_step["content"] = df[text_column].astype(str)
        st.success(f"Kolom '{text_column}' dipakai sebagai teks ulasan (disimpan sebagai 'content' internal).")
        show_df(st.session_state.df_step, "Preview Data (Awal)")

# =========================
# Resource (Kamus & Lexicon)
# =========================
st.subheader("Resource (Kamus & Lexicon)")

colr1, colr2, colr3 = st.columns([1, 1, 1])

with colr1:
    if st.button("Load Kamus Excel", disabled=(kamus_file is None)):
        try:
            st.session_state.kamus_dict = load_kamus_from_excel(kamus_file)
            st.success(f"Kamus loaded: {len(st.session_state.kamus_dict)} entri.")
        except Exception as e:
            st.error(f"Gagal load kamus: {e}")

with colr2:
    if st.button("Load Lexicon Positive", disabled=(lex_pos_file is None)):
        st.session_state.lex_pos = load_lexicon_csv(lex_pos_file)
        st.success(f"Lexicon positive loaded: {len(st.session_state.lex_pos)} entri.")

with colr3:
    if st.button("Load Lexicon Negative", disabled=(lex_neg_file is None)):
        st.session_state.lex_neg = load_lexicon_csv(lex_neg_file)
        st.success(f"Lexicon negative loaded: {len(st.session_state.lex_neg)} entri.")


# =========================
# STEP 1: Preprocessing bertahap + Filter Lexicon
# =========================
st.header("STEP 1 — Preprocessing (bertahap) + Filter Lexicon (hapus typo/OOV)")

if st.session_state.df_step is None:
    st.info("Klik tombol 'Gunakan kolom ini sebagai teks ulasan' pada STEP 0 dulu.")
else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        if st.button("1) Case Folding"):
            st.session_state.df_step["content"] = st.session_state.df_step["content"].apply(CaseFolding)
            st.success("Case folding selesai.")

    with c2:
        if st.button("2) Normalisasi Kamus", disabled=(st.session_state.kamus_dict is None)):
            st.session_state.df_step["content"] = st.session_state.df_step["content"].apply(
                lambda x: normalisasi_dengan_kamus(x, st.session_state.kamus_dict)
            )
            st.success("Normalisasi kamus selesai.")

    with c3:
        if st.button("3) Data Cleaning"):
            st.session_state.df_step["content"] = st.session_state.df_step["content"].apply(datacleaning)
            st.success("Data cleaning selesai.")

    with c4:
        if st.button("4) Stopword Removal"):
            st.session_state.df_step["content"] = st.session_state.df_step["content"].apply(remove_stopwords)
            st.success("Stopword removal selesai.")

    with c5:
        if st.button("5) Stemming (Sastrawi)"):
            st.session_state.df_step["content"] = st.session_state.df_step["content"].apply(stem_text)
            st.success("Stemming selesai.")

    with c6:
        if st.button("6) Filter Lexicon", disabled=(st.session_state.lex_pos is None or st.session_state.lex_neg is None)):
            dfp = st.session_state.df_step.copy()
            dfp["content_list"] = dfp["content"].astype(str).str.split()
            dfp["content_list"] = dfp["content_list"].apply(
                lambda toks: filter_tokens_by_lexicon(toks, st.session_state.lex_pos, st.session_state.lex_neg)
            )
            dfp["content"] = dfp["content_list"].apply(lambda toks: " ".join(toks))
            st.session_state.df_step = dfp
            st.success("Filter lexicon selesai. Kata di luar lexicon dihapus (typo/OOV dibuang).")

    col_clean_a, col_clean_b = st.columns([1, 1])
    with col_clean_a:
        if st.button("Bersihkan baris kosong/NaN setelah preprocessing"):
            dfp = st.session_state.df_step.copy()
            dfp["content"] = dfp["content"].fillna("").astype(str)
            dfp = dfp[dfp["content"].str.strip() != ""]
            dfp = dfp.dropna(subset=["content"]).reset_index(drop=True)
            st.session_state.df_step = dfp
            st.success("Baris kosong/NaN dihapus.")

    with col_clean_b:
        if st.button("Reset ke data awal (STEP 0)"):
            if st.session_state.raw_df is not None and st.session_state.chosen_text_column in st.session_state.raw_df.columns:
                st.session_state.df_step = pd.DataFrame()
                st.session_state.df_step["content"] = st.session_state.raw_df[st.session_state.chosen_text_column].astype(str)
                # reset model state juga
                st.session_state.tfidf = None
                st.session_state.svm = None
                st.session_state.report = None
                st.session_state.cm = None
                st.success("Data direset ke kondisi awal.")
            else:
                st.warning("Data awal belum siap. Pastikan STEP 0 sudah dilakukan.")

    show_df(st.session_state.df_step, "Preview Setelah Preprocessing")


# =========================
# STEP 2: Lexicon labeling
# =========================
st.header("STEP 2 — Lexicon-based Sentiment Labeling")

if st.session_state.df_step is None:
    st.info("Load dataset dulu.")
elif st.session_state.lex_pos is None or st.session_state.lex_neg is None:
    st.info("Load lexicon positive & negative dulu.")
else:
    if st.button("Jalankan Lexicon Labeling"):
        df = st.session_state.df_step.copy()

        # pakai content_list yang sudah ada dari filter lexicon jika ada
        if "content_list" not in df.columns:
            df["content_list"] = df["content"].astype(str).str.split()

        results = df["content_list"].apply(
            lambda toks: sentiment_analysis_lexicon_indonesia(toks, st.session_state.lex_pos, st.session_state.lex_neg)
        )
        results = list(zip(*results))
        df["score"] = results[0]
        df["Sentimen"] = results[1]

        st.session_state.df_step = df
        st.success("Lexicon labeling selesai.")

    if st.session_state.df_step is not None and "Sentimen" in st.session_state.df_step.columns:
        show_df(st.session_state.df_step, "Preview Setelah Labeling")
        st.write("Distribusi Sentimen (Lexicon):")
        st.write(st.session_state.df_step["Sentimen"].value_counts())

        if st.button("Filter netral (score == 0)"):
            df = st.session_state.df_step.copy()
            df = df[df["score"] != 0].reset_index(drop=True)
            st.session_state.df_step = df
            st.success("Netral dihapus (score != 0).")
            st.write("Distribusi Sentimen (setelah filter):")
            st.write(st.session_state.df_step["Sentimen"].value_counts())


# =========================
# STEP 3: TF-IDF
# =========================
st.header("STEP 3 — TF-IDF Feature Extraction")

if st.session_state.df_step is None:
    st.info("Load dataset dulu.")
elif "Sentimen" not in st.session_state.df_step.columns:
    st.info("Lakukan lexicon labeling dulu (STEP 2).")
else:
    if st.button("Fit TF-IDF pada data"):
        df = st.session_state.df_step.copy()

        if "content_list" not in df.columns:
            df["content_list"] = df["content"].astype(str).str.split()

        X_text = df["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

        tfidf = TfidfVectorizer()
        tfidf.fit(X_text)

        st.session_state.tfidf = tfidf
        st.success(f"TF-IDF fitted. Jumlah fitur: {len(tfidf.get_feature_names_out())}")

    if st.session_state.tfidf is not None:
        st.write("Contoh feature names (20 pertama):")
        st.write(st.session_state.tfidf.get_feature_names_out()[:20])


# =========================
# STEP 4: Train/Test Split + SVM
# =========================
st.header("STEP 4 — Train/Test Split & Train SVM")

if st.session_state.tfidf is None:
    st.info("Fit TF-IDF dulu (STEP 3).")
else:
    if st.button("Split data & Train SVM"):
        df = st.session_state.df_step.copy()

        # pastikan hanya negatif/positif
        if "Sentimen" not in df.columns:
            st.error("Kolom 'Sentimen' belum ada. Jalankan STEP 2 dulu.")
        else:
            df = df[df["Sentimen"].isin(["negatif", "positif"])].reset_index(drop=True)
            if df.empty:
                st.error("Tidak ada data negatif/positif untuk training. Pastikan sudah filter netral (opsional) dan ada data.")
            else:
                if "content_list" not in df.columns:
                    df["content_list"] = df["content"].astype(str).str.split()

                X_text = df["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
                y = df["Sentimen"]

                X_tfidf = st.session_state.tfidf.transform(X_text).toarray()

                X_train, X_test, y_train, y_test = train_test_split(
                    X_tfidf, y, test_size=float(test_size), random_state=int(random_state)
                )

                svm_model = SVC(kernel=kernel, C=float(C))
                svm_model.fit(X_train, y_train)

                y_pred = svm_model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.session_state.svm = svm_model
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
                st.session_state.y_train, st.session_state.y_test = y_train, y_test
                st.session_state.y_pred = y_pred

                st.success(f"SVM trained. Accuracy: {acc:.4f}")

                st.session_state.report = classification_report(y_test, y_pred, zero_division=0)
                st.session_state.cm = confusion_matrix(y_test, y_pred, labels=["negatif", "positif"])

    if st.session_state.report is not None:
        st.subheader("Classification Report")
        st.code(st.session_state.report)

    if st.session_state.cm is not None:
        st.subheader("Confusion Matrix")
        plot_confusion(st.session_state.cm, labels=("negatif", "positif"), title="Confusion Matrix SVM")


# =========================
# STEP 5: Prediksi pada dataset ini + Export
# =========================
st.header("STEP 5 — Prediksi (opsional) & Export Hasil")

if st.session_state.df_step is None:
    st.info("Belum ada data.")
else:
    if st.button("Tambahkan kolom prediksi SVM", disabled=(st.session_state.svm is None or st.session_state.tfidf is None)):
        df = st.session_state.df_step.copy()

        if "content_list" not in df.columns:
            df["content_list"] = df["content"].astype(str).str.split()

        X_text = df["content_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
        X_tfidf = st.session_state.tfidf.transform(X_text).toarray()

        df["Prediksi_SVM"] = st.session_state.svm.predict(X_tfidf)
        st.session_state.df_step = df
        st.success("Prediksi SVM ditambahkan ke dataframe.")

    show_df(st.session_state.df_step, "Preview Data Final")

    csv_bytes = st.session_state.df_step.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download hasil sebagai CSV",
        data=csv_bytes,
        file_name="hasil_sentiment_streamlit.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Deploy: pastikan repo berisi app.py + requirements.txt. Streamlit Cloud akan auto-install dependency dari requirements.txt.")

