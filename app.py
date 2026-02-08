import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import io
import csv
import pickle
import time

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
# UI STYLE — CLEAN & AKADEMIK
# =========================================================
def inject_academic_style():
    st.markdown("""
    <style>
    .stApp { background-color:#0b0f17; color:#e5e7eb; }
    section[data-testid="stSidebar"] {
        background:#0a0e16;
        border-right:1px solid rgba(255,255,255,0.06);
    }
    h1,h2,h3,h4 { color:#f9fafb; }
    .stButton>button {
        background:#1f2937; color:white;
        border:1px solid rgba(255,255,255,0.12);
        border-radius:8px;
        font-weight:600;
    }
    .stDownloadButton>button {
        background:#111827; color:white;
        border:1px solid rgba(255,255,255,0.18);
        border-radius:8px;
        font-weight:700;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# NLTK SETUP
# =========================================================
@st.cache_resource
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

ensure_nltk()


# =========================================================
# LOAD RESOURCE DARI REPOSITORY
# =========================================================
KAMUS_PATH = "kamuskatabaku (1).xlsx"
LEX_POS_PATH = "positive.csv"
LEX_NEG_PATH = "negative.csv"

@st.cache_resource
def load_kamus():
    df = pd.read_excel(KAMUS_PATH)
    return dict(zip(df["non_standard"], df["standard_word"]))

@st.cache_resource
def load_lexicon(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                lex[row[0]] = int(row[1])
    return lex


# =========================================================
# PREPROCESSING FUNCTIONS
# =========================================================
def case_folding(text): return str(text).lower()

def normalisasi(text, kamus):
    return " ".join([kamus.get(w, w) for w in text.split()])

def cleaning(text):
    text = re.sub(r"http\S+|[^a-z\s]", " ", text)
    return emoji.replace_emoji(text, "").strip()

def stopword_removal(text):
    stop = set(stopwords.words("indonesian"))
    return " ".join([w for w in text.split() if w not in stop])

@st.cache_resource
def stemmer():
    return StemmerFactory().create_stemmer()

def stemming(text): return stemmer().stem(text)

def filter_lexicon(tokens, pos, neg):
    allowed = set(pos) | set(neg)
    return [t for t in tokens if t in allowed]


def sentiment_lexicon(tokens, pos, neg):
    score = sum(pos.get(t,0)+neg.get(t,0) for t in tokens)
    if score > 0: return score, "positif"
    if score < 0: return score, "negatif"
    return score, "netral"


# =========================================================
# SESSION STATE
# =========================================================
if "menu" not in st.session_state: st.session_state.menu = "Home"


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config("Analisis Sentimen TF-IDF + SVM", layout="wide")
inject_academic_style()

kamus = load_kamus()
lex_pos = load_lexicon(LEX_POS_PATH)
lex_neg = load_lexicon(LEX_NEG_PATH)


# =========================================================
# SIDEBAR — NAVIGASI & PROGRESS
# =========================================================
def nav_button(label, icon):
    if st.sidebar.button(f"{icon} {label}"):
        st.session_state.menu = label

st.sidebar.title("Navigasi")
nav_button("Home", "🏠")
nav_button("Input", "📥")
nav_button("Preprocessing", "🧽")
nav_button("Klasifikasi SVM", "🤖")

st.sidebar.markdown("---")
st.sidebar.subheader("Progress")
st.sidebar.write("✅ Input" if "df" in st.session_state else "⬜ Input")
st.sidebar.write("✅ Preprocessing" if "df_pre" in st.session_state else "⬜ Preprocessing")
st.sidebar.write("✅ SVM" if "svm" in st.session_state else "⬜ SVM")


# =========================================================
# HOME
# =========================================================
if st.session_state.menu == "Home":
    st.title("Sistem Analisis Sentimen Ulasan")
    st.write("""
    Sistem ini menggunakan metode **TF-IDF** dan **Support Vector Machine (SVM)**  
    untuk melakukan analisis sentimen ulasan pengguna secara bertahap.
    """)
    if st.button("🚀 Mulai"):
        st.session_state.menu = "Input"


# =========================================================
# INPUT
# =========================================================
elif st.session_state.menu == "Input":
    st.title("Input Data")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        col = st.selectbox("Pilih kolom teks", df.columns)
        if st.button("Gunakan Data"):
            st.session_state.df = df[[col]].rename(columns={col:"content"})
            st.session_state.menu = "Preprocessing"


# =========================================================
# PREPROCESSING
# =========================================================
elif st.session_state.menu == "Preprocessing":
    st.title("Preprocessing")

    mode = st.radio(
        "Metode preprocessing:",
        ["Otomatis (semua tahap)", "Tahap per tahap"]
    )

    df = st.session_state.df.copy()

    if mode == "Otomatis (semua tahap)":
        if st.button("▶️ Jalankan Preprocessing"):
            with st.spinner("Memproses..."):
                df["content"] = df["content"].apply(case_folding)
                df["content"] = df["content"].apply(lambda x: normalisasi(x, kamus))
                df["content"] = df["content"].apply(cleaning)
                df["content"] = df["content"].apply(stopword_removal)
                df["content"] = df["content"].apply(stemming)
                df["tokens"] = df["content"].str.split()
                df["tokens"] = df["tokens"].apply(lambda x: filter_lexicon(x, lex_pos, lex_neg))
                df["content"] = df["tokens"].apply(lambda x: " ".join(x))
                df["score"], df["Sentimen"] = zip(*df["tokens"].apply(
                    lambda x: sentiment_lexicon(x, lex_pos, lex_neg)
                ))
            st.session_state.df_pre = df[df["score"]!=0]
            st.success("Preprocessing selesai")

    else:
        st.info("Gunakan tombol di setiap tahap (disederhanakan untuk kejelasan akademik).")

    if "df_pre" in st.session_state:
        st.dataframe(st.session_state.df_pre.head())
        st.download_button(
            "Download hasil preprocessing",
            data=io.BytesIO(st.session_state.df_pre.to_excel(index=False)),
            file_name="preprocessing.xlsx"
        )
        if st.button("➡️ Lanjut ke Klasifikasi"):
            st.session_state.menu = "Klasifikasi SVM"


# =========================================================
# KLASIFIKASI SVM
# =========================================================
elif st.session_state.menu == "Klasifikasi SVM":
    st.title("Klasifikasi SVM")

    df = st.session_state.df_pre
    X_text = df["content"]
    y = df["Sentimen"]

    st.subheader("Parameter Model")
    col1,col2,col3,col4 = st.columns(4)
    test_size = col1.slider("Test size",0.1,0.4,0.2)
    random_state = col2.number_input("Random state",0,42)
    C = col3.number_input("C",0.01,1.0)
    kernel = col4.selectbox("Kernel",["linear","rbf","poly"])

    if st.button("▶️ Jalankan Klasifikasi"):
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(X_text)

        Xtr,Xte,ytr,yte = train_test_split(
            X,y,test_size=test_size,random_state=random_state
        )

        svm = SVC(kernel=kernel,C=C)
        svm.fit(Xtr,ytr)
        ypred = svm.predict(Xte)

        st.session_state.svm = svm

        st.metric("Accuracy", accuracy_score(yte,ypred))
        st.code(classification_report(yte,ypred))

        cm = confusion_matrix(yte,ypred,labels=["negatif","positif"])
        fig,ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt="d",ax=ax)
        st.pyplot(fig)
