import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
# import joblib
import re
# from unidecode import unidecode
# from nltk.corpus import stopwords  
# import torch
# from transformers import BertTokenizer, BertModel

# ---------- Config ----------
st.set_page_config(
    page_title="Dashboard Proyectos de Ley",
    layout="wide"
)

# model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)

# def limpiar_texto(texto):
#     # Primera etapa: limpieza básica
#     texto = texto.lower()                             # a minúsculas
#     texto = unidecode(texto)                          # quitar tildes y normalizar
#     texto = re.sub(r'n°|ndeg', '', texto)             # eliminar "N°" o "ndeg"
#     texto = re.sub(r'\s+', ' ', texto)                # espacios duplicados → uno solo
#     texto = texto.strip()                             # eliminar espacios iniciales/finales

#     # Segunda etapa: eliminación de elementos no informativos
#     texto = re.sub(r'\d+', '', texto)                 # Eliminar números
#     texto = re.sub(r'[^\w\s]', '', texto)             # Eliminar puntuación
#     texto = re.sub(r'\s+', ' ', texto).strip()        # Espacios múltiples

#     # Tokenizar y filtrar palabras
#     palabras = texto.split()
#     palabras_filtradas = [
#         p for p in palabras
#         if p not in stopwords_es                      # stopwords estándar
#         and p not in stopwords_custom                 # stopwords personalizadas
#         and len(p) > 2                                # eliminar palabras muy cortas (<=2)
#     ]
#     return " ".join(palabras_filtradas)

# def extract_features(text):
#     # Tokenizar el texto (convertir el texto en tokens)
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

#     # Desactivar el cálculo de gradientes ya que solo estamos haciendo inferencia
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Extraer la última capa oculta de BERT (esto es lo que usaremos como características)
#     last_hidden_states = outputs.last_hidden_state

#     # Obtener el embedding del primer token [CLS] que representa todo el texto
#     cls_embedding = last_hidden_states[0][0].numpy()

#     return cls_embedding

# stopwords_es = set(stopwords.words('spanish'))
# stopwords_custom = set(['rt', '-', '...', '“', '”', '¿', '¡','N°'])
# tokenizer = BertTokenizer.from_pretrained(model_name)

@st.cache_data
def load_data():
    df_cnt = pd.read_excel("df_proyectosLey_cantComentariosPosNeg.xlsx", engine="openpyxl")
    df_cmt = pd.read_excel("comentarios_unido.xlsx", engine="openpyxl")
    df_cat = pd.read_excel("proyectosLeyes_unido.xlsx", engine="openpyxl")

    # Limpieza
    df_cnt.columns = df_cnt.columns.str.strip()
    df_cmt.columns = df_cmt.columns.str.strip()
    df_cat.columns = df_cat.columns.str.strip()

    
    # En load_data()
    if "Numero_Proyecto_Ley" in df_cmt.columns:
        df_cmt["Numero_Proyecto_Ley"] = df_cmt["Numero_Proyecto_Ley"].astype(str)
    if "Numero_Proyecto_Ley" in df_cnt.columns:
        df_cnt["Numero_Proyecto_Ley"] = df_cnt["Numero_Proyecto_Ley"].astype(str)

    # Si en el catálogo usas "Número de Expediente" como la misma llave, conviértela también:
    if "Número de Expediente" in df_cat.columns:
        df_cat["Número de Expediente"] = df_cat["Número de Expediente"].astype(str)


    # Fecha
    if "Fecha" in df_cmt.columns:
        df_cmt["Fecha"] = pd.to_datetime(df_cmt["Fecha"], errors="coerce", dayfirst=False, infer_datetime_format=True)
        df_cmt["Fecha_fmt"] = df_cmt["Fecha"].dt.strftime("%Y/%m/%d")

    # Etiqueta normalizada
    if "Etiqueta" in df_cmt.columns:
        df_cmt["Etiqueta"] = df_cmt["Etiqueta"].astype(str).str.upper().str.strip()

    return df_cnt, df_cmt, df_cat

df_cnt, df_cmt, df_cat = load_data()

# Sidebar - Filtros
st.sidebar.title("Filtros")



# Rango de fechas
if "Fecha" in df_cmt.columns and df_cmt["Fecha"].notna().any():
    min_date = pd.to_datetime(df_cmt["Fecha"].min())
    max_date = pd.to_datetime(df_cmt["Fecha"].max())
    date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date))
else:
    date_range = None


# Sidebar - Filtro de etiquetas (solo POSITIVO y NEGATIVO)
etiquetas = ["POSITIVO", "NEGATIVO"]  # opciones fijas
sel_etiquetas = st.sidebar.multiselect(
    "Etiqueta de comentario",
    options=etiquetas,
    default=etiquetas
)


# (Opcionales: se activarán luego de merge con catálogo)
grupos = sorted(df_cat["Grupo Parlamentario"].dropna().unique().tolist()) if "Grupo Parlamentario" in df_cat.columns else []
sel_grupo = st.sidebar.multiselect("Grupo parlamentario", options=grupos, default=[])

estados = sorted(df_cat["Último Estado"].dropna().unique().tolist()) if "Último Estado" in df_cat.columns else []
sel_estado = st.sidebar.multiselect("Último estado", options=estados, default=[])

text_query = st.sidebar.text_input("Buscar por título / número de PL", "")

# Aplicar filtros a comentarios
df_cmt_f = df_cmt.copy()

if date_range and isinstance(date_range, tuple):
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_cmt_f = df_cmt_f[(df_cmt_f["Fecha"] >= start) & (df_cmt_f["Fecha"] <= end)]

if sel_etiquetas:
    df_cmt_f = df_cmt_f[df_cmt_f["Etiqueta"].isin(sel_etiquetas)]

# Texto: corregimos paréntesis y aplicamos a ambos DF
df_cnt_f = df_cnt.copy()
if text_query.strip():
    tq = text_query.strip().lower()
    df_cnt_f = df_cnt[
        df_cnt["Proyecto_Ley"].str.lower().str.contains(tq, na=False) |
        df_cnt["Numero_Proyecto_Ley"].str.lower().str.contains(tq, na=False)
    ]
    df_cmt_f = df_cmt_f[
        df_cmt_f["Numero_Proyecto_Ley"].astype(str).str.lower().str.contains(tq, na=False) |
        df_cmt_f["Proyecto_Ley"].str.lower().str.contains(tq, na=False)
    ]



# --- Merge comentarios + catálogo ---
df_all = df_cmt_f.merge(
    df_cat[["Número de Expediente", "Grupo Parlamentario", "Último Estado", "Título"]],
    left_on="Numero_Proyecto_Ley",
    right_on="Número de Expediente",
    how="left"
)

# --- Filtros del catálogo ---
if sel_grupo:
    df_all = df_all[df_all["Grupo Parlamentario"].isin(sel_grupo)]
if sel_estado:
    df_all = df_all[df_all["Último Estado"].isin(sel_estado)]

# --- Normalización ---
df_all["Etiqueta"] = df_all["Etiqueta"].astype(str).str.upper().str.strip()
df_all["Numero_Proyecto_Ley"] = df_all["Numero_Proyecto_Ley"].astype(str)

# --- MÉTRICAS ARRIBA ---
proyectos_con_coment = df_all["Numero_Proyecto_Ley"].nunique()
total_comentarios = len(df_all)
pos_count = (df_all["Etiqueta"] == "POSITIVO").sum()
neg_count = (df_all["Etiqueta"] == "NEGATIVO").sum()
pos_pct = (pos_count / total_comentarios * 100) if total_comentarios else 0
neg_pct = (neg_count / total_comentarios * 100) if total_comentarios else 0

st.title("📊 Dashboard de Proyectos de Ley y Comentarios")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Proyectos con comentarios", f"{proyectos_con_coment:,}")
col2.metric("Comentarios", f"{total_comentarios:,}")
col3.metric("Positivos (%)", f"{pos_pct:,.1f}%")
col4.metric("Negativos (%)", f"{neg_pct:,.1f}%")
#st.markdown("---")


# ==========================
# MÉTRICAS DINÁMICAS (después de filtros)
# ==========================
# df_cmt_f: comentarios filtrados (por fecha, etiquetas, texto, etc.)
# df_cnt_f: conteos/proyectos filtrados por búsqueda (si aplicaste text_query)

# Proyectos con al menos un comentario en el conjunto filtrado
if "Numero_Proyecto_Ley" in df_cmt_f.columns:
    proyectos_con_coment = df_cmt_f["Numero_Proyecto_Ley"].nunique()
else:
    # Fallback por si en algún escenario no existe la columna
    proyectos_con_coment = df_cnt_f["Numero_Proyecto_Ley"].nunique()

# Total de comentarios del conjunto filtrado
total_comentarios = int(len(df_cmt_f))

# Conteo por etiqueta (según filtros activos)
# Normaliza etiqueta por seguridad (mayúsculas y sin espacios)
if "Etiqueta" in df_cmt_f.columns:
    df_cmt_f["Etiqueta"] = df_cmt_f["Etiqueta"].astype(str).str.upper().str.strip()
    pos_count = int((df_cmt_f["Etiqueta"] == "POSITIVO").sum())
    neg_count = int((df_cmt_f["Etiqueta"] == "NEGATIVO").sum())
else:
    pos_count, neg_count = 0, 0

# Porcentajes (protege división por cero)
pos_pct = (pos_count / total_comentarios * 100) if total_comentarios > 0 else 0.0

st.markdown("---")

# --------------------------------------------
# Encabezado de etiquetas si hay un solo proyecto
# --------------------------------------------
proyectos_unicos = df_cmt_f["Numero_Proyecto_Ley"].dropna().unique().tolist() if "Numero_Proyecto_Ley" in df_cmt_f.columns else []
if len(proyectos_unicos) == 1:
    unico_pl = proyectos_unicos[0]
    titulo_pl = df_cmt_f.loc[df_cmt_f["Numero_Proyecto_Ley"] == unico_pl, "Proyecto_Ley"].dropna().iloc[0] \
        if "Proyecto_Ley" in df_cmt_f.columns and (df_cmt_f["Numero_Proyecto_Ley"] == unico_pl).any() else f"Proyecto {unico_pl}"

    agg_etq = (
        df_cmt_f[df_cmt_f["Numero_Proyecto_Ley"] == unico_pl]
        .groupby("Etiqueta")
        .size()
        .reset_index(name="Comentarios")
        .sort_values("Comentarios", ascending=False)
    )
    total_pl = int(agg_etq["Comentarios"].sum())

    st.markdown("### 🏷️ Etiquetas del proyecto seleccionado")
    st.markdown(f"**{titulo_pl}**  \n*Número de Proyecto de Ley:* `{unico_pl}`")

    n = len(agg_etq)
    cols_chip = st.columns(n if n > 0 else 1)
    for i, row in enumerate(agg_etq.itertuples(index=False)):
        etiqueta = row.Etiqueta
        count = int(row.Comentarios)
        pct = (count / total_pl * 100) if total_pl else 0.0
        color = "#1f77b4" if etiqueta == "POSITIVO" else "#d62728"

        with cols_chip[i]:
            st.markdown(
                f"""
                <div style="
                    display:inline-block;
                    padding:6px 10px;
                    border-radius:16px;
                    background-color:{color};
                    color:white;
                    font-weight:600;
                    margin-bottom:6px;">
                    {etiqueta}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.metric("Comentarios", f"{count:,}", help=f"{pct:,.1f}% del total del proyecto")
    st.markdown("---")


# Antes de construir el gráfico:
df_cmt_f["Numero_Proyecto_Ley"] = df_cmt_f["Numero_Proyecto_Ley"].astype(str)
# --------------------------------------------
# Gráfico: Positivos vs Negativos por Número de Proyecto de Ley (Top 20)
# --------------------------------------------
if {"Numero_Proyecto_Ley", "Proyecto_Ley", "Etiqueta"}.issubset(df_cmt_f.columns):
    # Normalizar etiqueta por seguridad
    df_cmt_f["Etiqueta"] = df_cmt_f["Etiqueta"].astype(str).str.upper().str.strip()

    # 1) Agregado
    agg = (
        df_cmt_f
        .groupby(["Numero_Proyecto_Ley", "Proyecto_Ley", "Etiqueta"])
        .size()
        .reset_index(name="Comentarios")
    )

    if agg.empty:
        st.warning("No hay datos para el filtro aplicado (no se puede graficar).")
    else:
        # 2) Pivot para conocer el total por proyecto y ordenar
        pivot = (
            agg.pivot_table(
                index=["Numero_Proyecto_Ley", "Proyecto_Ley"],
                columns="Etiqueta",
                values="Comentarios",
                fill_value=0
            )
            .reset_index()
        )

        # Asegurar columnas (evita KeyError cuando filtras una sola etiqueta)
        for col in ["POSITIVO", "NEGATIVO"]:
            if col not in pivot.columns:
                pivot[col] = 0

        # 3) Total por proyecto y Top 20
        pivot["TotalComentarios"] = pivot["POSITIVO"] + pivot["NEGATIVO"]
        pivot = pivot.sort_values("TotalComentarios", ascending=False).head(20)

        # 4) Formato largo para stacked
        long_df = pivot.melt(
            id_vars=["Numero_Proyecto_Ley", "Proyecto_Ley"],
            value_vars=["POSITIVO", "NEGATIVO"],
            var_name="Etiqueta",
            value_name="Comentarios"
        )

        # 5) Gráfico — eje X: Numero_Proyecto_Ley, color: Etiqueta (POS/NEG)
        fig_bar = px.bar(
            long_df,
            x="Numero_Proyecto_Ley",          # <- eje X por número de proyecto
            y="Comentarios",
            color="Etiqueta",                  # <- POSITIVO / NEGATIVO
            hover_data=["Proyecto_Ley", "Numero_Proyecto_Ley"],
            title="Comentarios por N° de Proyecto de Ley (Top 20) – Positivos vs Negativos",
        )
        # Ordenar categorías X por total descendente usando datos agregados
        
        fig_bar.update_layout(
            xaxis_title="Número de Proyecto de Ley",
            yaxis_title="Comentarios",
            legend_title="Etiqueta",
            xaxis_type="category",                  # ← fuerza categórico
            xaxis_tickangle=-45,                    # ← opcional: mejor lectura
        )

        st.plotly_chart(fig_bar, use_container_width=True)


# --------------------------------------------
# Serie temporal (puedes elegir por Etiqueta o por Proyecto)
# --------------------------------------------
if "Fecha" in df_cmt_f.columns:
    # Opción 1: por Etiqueta (más clara para sentimiento)
    ts = (
        df_cmt_f
        .groupby([pd.Grouper(key="Fecha", freq="D"), "Etiqueta"])
        .size()
        .reset_index(name="Comentarios")
    )
    fig_ts = px.line(ts, x="Fecha", y="Comentarios", color="Etiqueta",
                     title="Evolución diaria de comentarios por etiqueta")
    st.plotly_chart(fig_ts, use_container_width=True)

    # # Opción 2: por Proyecto (deja esta si la prefieres)
    # ts_proj = (
    #     df_cmt_f
    #     .groupby([pd.Grouper(key="Fecha", freq="D"), "Proyecto_Ley"])
    #     .size()
    #     .reset_index(name="Comentarios")
    # )
    # fig_ts_proj = px.line(ts_proj, x="Fecha", y="Comentarios", color="Proyecto_Ley",
    #                       title="Evolución diaria de comentarios por Proyecto de Ley")
    # st.plotly_chart(fig_ts_proj, use_container_width=True)

# --------------------------------------------
# Tabla de comentarios (detalle)
# --------------------------------------------
st.subheader("🗂️ Detalle de comentarios")

# Aseguramos que Numero_Proyecto_Ley sea string
df_cmt_f["Numero_Proyecto_Ley"] = df_cmt_f["Numero_Proyecto_Ley"].astype(str)

# Columnas a mostrar (incluyendo Fecha_fmt)
cols_to_show = ["Numero_Proyecto_Ley", "Autor", "Etiqueta", "Fecha_fmt", "Comentario"]
cols_to_show = [c for c in cols_to_show if c in df_cmt_f.columns]

# Ordenar por Fecha_fmt (string YYYY/MM/DD)
df_show = df_cmt_f[cols_to_show].sort_values("Fecha_fmt", ascending=False)

# Mostrar tabla
st.dataframe(df_show, use_container_width=True)



# st.markdown("---")
# st.subheader("ℹ️ Catálogo de proyectos (sincronizar llave)")
# st.info("""
# Selecciona un proyecto en los gráficos o filtra por texto para ubicarlo.
# Luego definimos la relación exacta (llave) para enlazar con el catálogo
# y mostrar ficha completa (proponente, comisiones, último estado, etc.).
# """)


# import streamlit as st
# import joblib
# import numpy as np

# st.title("🤖 Prueba de Modelo BERT + RandomForest")

# # Cargar el modelo entrenado
# @st.cache_resource
# def load_model():
#     return joblib.load("modelo_bert_randomforest_optimo.pkl")

# model = load_model()

# # Entrada del usuario
# texto = st.text_area("Escribe el texto del proyecto de ley:", "")

# if st.button("Predecir Sentimiento"):
#     if texto.strip():
#         # Limpieza del texto
#         comentario_limpio = limpiar_texto(texto)

#         # Obtener embedding BERT
#         vector_comentario = extract_features(comentario_limpio)

#         # Validar que el embedding no sea None
#         if vector_comentario is None or len(vector_comentario) == 0:
#             st.error("No se pudo generar el embedding para el texto.")
#         else:
#             # Convertir a array 2D para predict
#             vector_comentario = np.array(vector_comentario).reshape(1, -1)

#             # Predicción
#             pred = model.predict(vector_comentario)[0]

#             # Mostrar resultado
#             if pred == "POSITIVO":
#                 st.success("✅ Predicción: POSITIVO")
#             elif pred == "NEGATIVO":
#                 st.error("❌ Predicción: NEGATIVO")
#             else:
#                 st.info(f"Predicción: {pred}")
#     else:
#         st.warning("Por favor, ingresa un texto para predecir.")


# --- Navegación a la página de predicción ---
with st.sidebar:
    st.markdown("---")
    st.page_link(
        "pages/02_Prediccion_modelo.py",
        label="🔮 Predicción con modelo .pkl",
        icon=":material/auto_awesome:",
        help="Ir a la 2da página"
    )


