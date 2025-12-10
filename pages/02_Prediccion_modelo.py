
# pages/02_Prediccion_modelo.py
import streamlit as st
import pandas as pd
import requests
import numpy as np
import joblib
import re
from io import BytesIO
import os
from sklearn.metrics.pairwise import cosine_similarity

# Librerías para limpieza/embeddings
import nltk
from unidecode import unidecode
import torch
from transformers import AutoTokenizer, AutoModel  # Auto* es más robusto que clases específicas

st.title("🔮 Predicción usando embeddings BETO")

# --------------------------
# Cache de recursos (Streamlit 1.18+). Fallback para versiones antiguas.
# --------------------------
CACHE_RESOURCE = getattr(st, "cache_resource", None) or getattr(st, "experimental_singleton", st.cache)
CACHE_DATA     = getattr(st, "cache_data", None)     or getattr(st, "experimental_memo", st.cache)

@CACHE_DATA
def ensure_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    from nltk.corpus import stopwords
    stop_es = set(stopwords.words("spanish"))
    # Personalizadas
    stop_custom = set(["rt", "-", "...", "“", "”", "¿", "¡", "N°"])
    return stop_es, stop_custom

stopwords_es, stopwords_custom = ensure_stopwords()

@CACHE_RESOURCE
def load_bert(model_name: str = "dccuchile/bert-base-spanish-wwm-uncased"):
    """
    Carga tokenizer y modelo BETO (Spanish BERT).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.eval()
    return tokenizer, bert_model

@CACHE_RESOURCE
def load_model_from_bytes(file_bytes: bytes):
    return joblib.load(BytesIO(file_bytes))

@CACHE_RESOURCE
def load_model_from_path(path: str):
    if path.startswith("http"):
        # Descargar el archivo desde la URL
        response = requests.get(path)
        if response.status_code != 200:
            raise FileNotFoundError(f"No se pudo descargar el archivo desde: {path}")
        # Cargar el modelo desde memoria
        return joblib.load(io.BytesIO(response.content))
    else:
        # Ruta local
        return joblib.load(path)


# --------------------------
# Limpieza (similar a tu entrenamiento)
# --------------------------
def limpiar_texto(texto: str) -> str:
    # Si viene como lista de tokens, únelos
    if isinstance(texto, list):
        texto = " ".join(texto)

    # Primera etapa: normalización básica
    texto = str(texto).lower()
    texto = unidecode(texto)
    texto = re.sub(r"n°|\bndeg\b", "", texto)           # eliminar "N°"/"ndeg"
    texto = re.sub(r"\s+", " ", texto).strip()          # espacios múltiples

    # Segunda etapa: eliminar números y puntuación
    texto = re.sub(r"\d+", "", texto)                   # números
    texto = re.sub(r"[^\w\s]", "", texto)               # puntuación
    texto = re.sub(r"\s+", " ", texto).strip()

    # Filtrar stopwords y tokens cortos
    palabras = texto.split()
    palabras_filtradas = [
        p for p in palabras
        if p not in stopwords_es
        and p not in stopwords_custom
        and len(p) > 2
    ]
    return " ".join(palabras_filtradas)

# --------------------------
# Embeddings BETO: vector [CLS] (768 dim en BERT-base)
# --------------------------
def extract_features(texto: str, tokenizer, bert_model) -> np.ndarray:
    texto = limpiar_texto(texto)
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # last_hidden_state: [batch, seq_len, hidden_size]
        # Tomamos el vector del primer token ([CLS])
        cls_vec = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]  # shape (768,)
    return cls_vec

# --------------------------
# Predicción con .pkl entrenado sobre embeddings
# --------------------------
def normalizar_etiqueta(y):
    if isinstance(y, str):
        y = y.strip().upper()
        if y in ("POS", "POSITIVE"):
            return "POSITIVO"
        if y in ("NEG", "NEGATIVE"):
            return "NEGATIVO"
        return y
    if y in (1, "1"):
        return "POSITIVO"
    if y in (0, "0"):
        return "NEGATIVO"
    return str(y)

def predecir_texto_con_embeddings(clf_model, texto, tokenizer, bert_model):
    vec = extract_features(texto, tokenizer, bert_model)       # (768,)
    X = np.array(vec).reshape(1, -1)                           # (1, 768)
    y_pred = clf_model.predict(X)[0]
    proba_fn = getattr(clf_model, "predict_proba", None)
    conf = None
    if callable(proba_fn):
        conf = float(np.max(proba_fn(X)))
    return normalizar_etiqueta(y_pred), conf

def predecir_lote_con_embeddings(clf_model, textos, tokenizer, bert_model):
    feats = [extract_features(t, tokenizer, bert_model) for t in textos]  # lista de (768,)
    X = np.vstack(feats)  # (n, 768)
    y_preds = clf_model.predict(X)
    proba_fn = getattr(clf_model, "predict_proba", None)
    confs = None
    if callable(proba_fn):
        confs = np.max(proba_fn(X), axis=1).tolist()
    y_norm = [normalizar_etiqueta(y) for y in y_preds]
    return y_norm, confs

# --------------------------
# Cargar BETO + clasificador .pkl
# --------------------------

# Cachear la carga de datos
@st.cache_data(show_spinner=True)
def load_data_final(path: str):
    # Si la ruta comienza con http, leer desde URL
    if path.startswith("http"):
        response = requests.get(path)
        if response.status_code != 200:
            raise FileNotFoundError(f"No se pudo descargar el archivo desde: {path}")
        return pd.read_csv(io.StringIO(response.text))
    else:
        return pd.read_csv(path)


# Cachear el cálculo de embeddings para todo el dataset
@st.cache_data(show_spinner=True)
def compute_dataset_embeddings(df: pd.DataFrame,
                               texto_col_preferida: str,
                               _tokenizer=None, _bert_model=None):
    """
    Devuelve:
    - df_valid: DataFrame con filas válidas (embedding generado)
    - E: matriz de embeddings (n_valid, 768)
    - texto_col_usada: nombre de la columna de texto utilizada
    """
    # Detectar columna de texto
    texto_col = texto_col_preferida
    if texto_col not in df.columns:
        # fallback a otras convenciones
        for cand in ["Comentario", "Texto", "texto", "comentario", "Contenido"]:
            if cand in df.columns:
                texto_col = cand
                break
    if texto_col not in df.columns:
        raise KeyError("No se encontró una columna de texto (Intentado: 'Texto', 'Comentario', etc.).")

    # Limpiar textos
    df = df.copy()
    df["Texto_Limpio"] = df[texto_col].fillna("").astype(str).apply(limpiar_texto)

    # Embeddings en lote
    feats = []
    idx_valid = []
    for i, t in enumerate(df["Texto_Limpio"]):
        try:
            v = extract_features(t, _tokenizer, _bert_model)  # (768,)
            if isinstance(v, np.ndarray) and v.size == 768:
                feats.append(v)
                idx_valid.append(i)
        except Exception:
            # descartar filas problemáticas
            pass

    if len(idx_valid) == 0:
        # No hay embeddings válidos
        df_valid = df.iloc[[]].copy()
        E = np.empty((0, 768))
        return df_valid, E, texto_col

    df_valid = df.iloc[idx_valid].copy()
    E = np.vstack(feats)  # (n_valid, 768)

    return df_valid, E, texto_col

tokenizer, bert_model = load_bert()   # BETO (Spanish BERT)
RUTA_PKL_FIJA = "https://raw.githubusercontent.com/BennyJCM/Dashboard-Proyectos-de-Ley/main/pages/modelo_bert_randomforest_optimo.pkl"

# Sustituye el text_input por ruta fija (raw string para Windows)
CSV_PATH = "https://raw.githubusercontent.com/BennyJCM/Dashboard-Proyectos-de-Ley/main/pages/data_final.csv"

# Carga cacheada del CSV (se hace al entrar a la página)
df_final = load_data_final(CSV_PATH)

# Prepara y cachea embeddings del dataset inmediatamente
df_valid, E, texto_col_usada = compute_dataset_embeddings(
    df_final,
    texto_col_preferida="Texto",  # cámbialo si tu columna real es distinta
    _tokenizer=tokenizer,
    _bert_model=bert_model
)



try:
    clf_model = load_model_from_path(RUTA_PKL_FIJA)
except Exception as e:
    st.error(f"No se pudo cargar el modelo .pkl desde la ruta fija.\n{e}")
    st.stop()



# ---------------------------------------
# Lectura automática de data_final.csv + similitud
# ---------------------------------------
ruta_csv = CSV_PATH  # Usa la ruta fija definida antes
# ---------------------------------------
# UI principal de predicción + similitud usando el dataset local
# ---------------------------------------
st.subheader("🔮 Predicción y Top‑N Proyectos de leyes más similares")
top_n = st.slider("Top‑N similares", min_value=5, max_value=50, value=5, step=5, key="topn_slider")

# Entrada del usuario (comentario del proyecto de ley)
texto_topn = st.text_area("Escribe el texto del proyecto de ley:", "", height=160, key="topn_text")

# Botón: al presionar, predice y muestra tabla de más similares
if st.button("Predecir y mostrar más similares", key="topn_btn"):
    if not texto_topn.strip():
        st.warning("Por favor, ingresa un texto para evaluar.")
        st.stop()

    # 1) Cargar data_final.csv
    try:
        df_final = load_data_final(ruta_csv)
    except Exception as e:
        st.exception(e)
        st.stop()

    # 2) Preparar embeddings del dataset (cacheado)
    try:
        df_valid, E, texto_col_usada = compute_dataset_embeddings(
            df_final,
            texto_col_preferida="Texto",  # tu preferencia por defecto
            _tokenizer=tokenizer,
            _bert_model=bert_model
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    if E.shape[0] == 0:
        st.warning("No se pudieron generar embeddings válidos para el dataset.")
        st.stop()

    
    # 3) Embedding del comentario ingresado (no recargamos el CSV aquí)
    vec_com = extract_features(texto_topn, tokenizer, bert_model)  # (768,)

    # 4) Calcular similitud coseno contra 'E' precomputado
    sims = cosine_similarity(vec_com.reshape(1, -1), E)[0]
    df_show = df_valid.copy()
    df_show["similaridad"] = sims


    # 5) Seleccionar columnas clave si existen
    cols_base = []
    for col in ["Numero_Proyecto_Ley", "Título", "Etiqueta"]:
        if col in df_valid.columns:
            cols_base.append(col)
    # Asegurar columna de texto usada y la similitud
    cols_finales = cols_base + [texto_col_usada, "similaridad"]

    # 6) Ordenar por mayor similitud y mostrar Top‑N
    top_df = df_show.sort_values("similaridad", ascending=False)[cols_finales].head(top_n)

    # 7) Predicción del `.pkl` con el embedding del comentario
    try:
        y_pred, conf = predecir_texto_con_embeddings(clf_model, texto_topn, tokenizer, bert_model)
    except Exception as e:
        st.exception(e)
        y_pred, conf = None, None

    st.markdown("### 🧠 Predicción del proyecto de Ley")
    if y_pred == "POSITIVO" and conf > 0.6:
        st.success(f"✅ Predicción: {y_pred}" + (f" (predicción: {conf-0.1:.2f})" if conf is not None else ""))
    # elif y_pred == "NEGATIVO" and conf <= 0.6:
    else:
        # st.info("Predicción no disponible." if y_pred is None else f"Predicción: {y_pred}, {}")
        st.error(f"❌ Predicción: NEGATIVO" + (f" (predicción: {conf-0.1:.2f})" if conf is not None else ""))

    st.markdown("### 🏆 Proyectos de Leyes más similares")
    st.dataframe(top_df, use_container_width=True)






