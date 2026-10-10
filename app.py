
from pathlib import Path
from html import escape
import os

# Compatibilidad del modelo entrenado con TensorFlow 2.10
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import requests

from PIL import Image, ImageOps



# =============================================================================
# 1. CONFIGURACIÓN DE LA APLICACIÓN
# =============================================================================

st.set_page_config(
    page_title="DocFlow AI | Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASES = [
    "Carta", "Formulario", "Correo electrónico", "Manuscrito",
    "Publicidad", "Informe científico", "Publicación científica",
    "Especificación", "Carpeta", "Noticia", "Presupuesto",
    "Factura", "Presentación", "Cuestionario", "Currículum",
    "Memorándum"
]

TAMANO_IMG = (384, 384)
UMBRAL_CONFIANZA = 0.60

RESULTADOS = pd.DataFrame({
    "Modelo": ["EfficientNet-B0", "ResNet50"],
    "Accuracy": [70.93, 70.14],
    "Precision": [71.04, 70.65],
    "Recall": [70.85, 70.03],
    "F1-score": [70.66, 70.02],
    "Entrenamiento (min)": [60.20, 87.97],
    "Evaluación (min)": [2.76, 4.38]
})


# =============================================================================
# 2. DISEÑO VISUAL
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, rgba(30, 200, 210, 0.10), transparent 32%),
        radial-gradient(ellipse at 5% 45%, rgba(123, 92, 240, 0.08), transparent 38%),
        #081321;
}

section[data-testid="stSidebar"] {
    background: #0D1B2C;
    border-right: 1px solid #24364A;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(115deg, #132D44 0%, #102438 53%, #16243F 100%);
    border: 1px solid #29455A;
    border-radius: 22px;
    padding: 35px 40px;
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: rgba(20, 220, 205, 0.12);
    filter: blur(65px);
    right: -60px;
    top: -120px;
}

.hero-kicker {
    font-size: 0.76rem;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    color: #4DE2D1;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero-title {
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.1;
    color: #F4F8FF;
    letter-spacing: -1.8px;
}

.hero-title span {
    color: #4DE2D1;
}

.hero-description {
    color: #B5C6D9;
    font-size: 1.02rem;
    max-width: 750px;
    line-height: 1.7;
    margin-top: 14px;
}

.section-label {
    color: #4DE2D1;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin: 20px 0 7px;
}

.section-heading {
    color: #F3F7FD;
    font-size: 1.55rem;
    font-weight: 800;
    margin-bottom: 18px;
}

.metric-card {
    background: linear-gradient(145deg, #14293D, #102033);
    border: 1px solid #2A4255;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 13px;
    min-height: 122px;
}

.metric-card.accent {
    border-color: rgba(77, 226, 209, 0.55);
    background: linear-gradient(145deg, #123C44, #102B3A);
}

.metric-label {
    color: #A6BDD0;
    font-size: 0.84rem;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #F5FAFF;
    font-size: clamp(1.25rem, 2.2vw, 1.95rem);
    line-height: 1.2;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.metric-note {
    color: #94AFC3;
    font-size: 0.77rem;
    margin-top: 10px;
}

.info-panel {
    background: #102438;
    border: 1px solid #284053;
    border-radius: 15px;
    padding: 20px 23px;
    color: #B9CADB;
    line-height: 1.7;
    margin: 12px 0 22px;
}

.sidebar-brand {
    font-size: 1.7rem;
    font-weight: 800;
    color: #F4F8FF;
    letter-spacing: -0.8px;
}

.sidebar-brand span {
    color: #4DE2D1;
}

.sidebar-caption {
    color: #95B2C7;
    font-size: 0.77rem;
    letter-spacing: 1.2px;
    margin-top: 6px;
}

.sidebar-divider {
    height: 1px;
    background: #2B4052;
    margin: 24px 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    border-bottom: 1px solid #294154;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 700;
    padding: 12px 5px;
}

.stTabs [aria-selected="true"] {
    color: #4DE2D1 !important;
}

div[data-testid="stFileUploader"] section {
    border-radius: 15px;
    border: 1px dashed #52829A;
    background: #102338;
}

div[data-testid="stMetric"] {
    background: #13283A;
    border: 1px solid #2B4457;
    padding: 17px 20px;
    border-radius: 15px;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

.footer {
    color: #819AAF;
    font-size: 0.8rem;
    border-top: 1px solid #243B4F;
    margin-top: 40px;
    padding-top: 18px;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. FUNCIONES
# =============================================================================

@st.cache_resource
def cargar_modelo():
    ruta = (
        Path(__file__).parent
        / "modelo"
        / "docflow_efficientnet_b0"
    )

    modelo = tf.saved_model.load(str(ruta))
    return modelo.signatures["serving_default"]


def preparar_imagen_web(imagen):
    imagen = imagen.convert("RGB")

    imagen = ImageOps.pad(
        imagen,
        TAMANO_IMG,
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255)
    )

    arreglo = np.asarray(imagen, dtype=np.float32)
    return np.expand_dims(arreglo, axis=0)


def clasificar_documento(imagen):
    entrada = preparar_imagen_web(imagen)
    predecir = cargar_modelo()

    resultado = predecir(
        imagenes=tf.constant(entrada, dtype=tf.float32)
    )

    probabilidades = resultado["probabilidades"].numpy()[0]

    indice = int(np.argmax(probabilidades))
    categoria = CLASES[indice]
    confianza = float(probabilidades[indice])

    return categoria, confianza, probabilidades

# Generar recomendaciones mediante Gemini
def generar_recomendacion_gemini(categoria, confianza):

    prompt = f"""
    Eres un asistente especializado en gestión documental.

    Un modelo de visión artificial clasificó un documento
    con los siguientes resultados:

    Categoría: {categoria}
    Confianza: {confianza:.2%}

    Entrega una recomendación profesional, breve y clara
    para organizar y gestionar este tipo de documento.

    Consideraciones:
    - Propón tres acciones concretas para el usuario.
    - Si la confianza es menor al 60%, recomienda revisión manual.
    - No inventes información específica del documento.
    - No afirmes haber visto o leído la imagen original.
    - Utiliza un lenguaje natural y fácil de comprender.
    - Responde en español.
    """

    respuesta = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
        headers={
            "x-goog-api-key": st.secrets["GEMINI_API_KEY"],
            "Content-Type": "application/json"
        },
        json={
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1000
            }
        },
        timeout=45
    )

    respuesta.raise_for_status()

    datos = respuesta.json()
    candidatos = datos.get("candidates", [])

    if not candidatos:
        raise ValueError("Gemini no devolvió una recomendación.")

    partes = candidatos[0].get("content", {}).get("parts", [])

    recomendacion = "".join(
        parte.get("text", "")
        for parte in partes
    ).strip()

    if not recomendacion:
        raise ValueError("Gemini devolvió una respuesta vacía.")

    return recomendacion

def tarjeta(titulo, valor, detalle="", destacada=False):
    clase = "metric-card accent" if destacada else "metric-card"

    st.markdown(
        f"""
        <div class="{clase}">
            <div class="metric-label">{escape(str(titulo))}</div>
            <div class="metric-value">{escape(str(valor))}</div>
            <div class="metric-note">{escape(str(detalle))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def grafico_probabilidades(probabilidades, categoria_predicha):
    datos = pd.DataFrame({
        "categoria": CLASES,
        "probabilidad": probabilidades.astype(float)
    })

    grafico = (
        alt.Chart(datos)
        .mark_bar(cornerRadiusEnd=5, size=17)
        .encode(
            x=alt.X(
                "probabilidad:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title="Probabilidad")
            ),
            y=alt.Y(
                "categoria:N",
                sort=alt.SortField(
                    field="probabilidad",
                    order="descending"
                ),
                title=None
            ),
            color=alt.condition(
                alt.datum.categoria == categoria_predicha,
                alt.value("#4DE2D1"),
                alt.value("#526E8A")
            ),
            tooltip=[
                alt.Tooltip("categoria:N", title="Categoría"),
                alt.Tooltip(
                    "probabilidad:Q",
                    title="Probabilidad",
                    format=".2%"
                )
            ]
        )
        .properties(
            height=520,
            background="#102438"
        )
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#B9CADB",
            titleColor="#B9CADB",
            gridColor="#264055",
            labelFontSize=12
        )
    )

    return grafico


# =============================================================================
# 4. BARRA LATERAL
# =============================================================================

st.sidebar.markdown(
    '<div class="sidebar-brand">DOCFLOW<span>AI</span></div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-caption">DOCUMENT INTELLIGENCE PLATFORM</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-divider"></div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("**Modelo de clasificación**")
st.sidebar.write("EfficientNet-B0")

st.sidebar.markdown("**Documentos reconocidos**")
st.sidebar.write("16 categorías")

st.sidebar.markdown("**Conjunto de prueba**")
st.sidebar.write("39.999 imágenes")

st.sidebar.markdown(
    '<div class="sidebar-divider"></div>',
    unsafe_allow_html=True
)

st.sidebar.metric("Accuracy en prueba", "70,93%")

st.sidebar.info(
    "Prototipo académico. Las clasificaciones son orientativas "
    "y deben revisarse antes de tomar decisiones importantes."
)


# =============================================================================
# 5. ENCABEZADO
# =============================================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Document intelligence · Prototype</div>
        <div class="hero-title">DocFlow <span>AI</span></div>
        <div class="hero-description">
            Clasificación inteligente de documentos mediante visión
            por computador. Una plataforma diseñada para organizar
            información, analizar resultados y apoyar la gestión documental.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# 6. NAVEGACIÓN PRINCIPAL
# =============================================================================

tab_clasificar, tab_resultados, tab_ia = st.tabs([
    "Clasificar documento",
    "Rendimiento del modelo",
    "Asistente IA"
])

categoria_actual = None
confianza_actual = None


# =============================================================================
# 7. CLASIFICACIÓN DE DOCUMENTOS
# =============================================================================

with tab_clasificar:

    st.markdown(
        '<div class="section-label">01 / CLASIFICACIÓN</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Analiza un documento</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Carga una imagen para identificar su categoría "
        "y consultar las probabilidades del modelo."
    )

    archivo = st.file_uploader(
        "Selecciona un documento",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if archivo is not None:

        try:
            with Image.open(archivo) as imagen_original:
                imagen = imagen_original.convert("RGB")

            with st.spinner("Analizando el documento..."):
                categoria, confianza, probabilidades = clasificar_documento(
                    imagen
                )

            categoria_actual = categoria
            confianza_actual = confianza

            columna_imagen, columna_resultados = st.columns([1, 1.2])

            with columna_imagen:

                st.markdown("#### Documento cargado")

                vista_previa = imagen.copy()
                vista_previa.thumbnail(
                    (850, 560),
                    Image.Resampling.LANCZOS
                )

                st.image(
                    vista_previa,
                    caption=archivo.name,
                    width="stretch"
                )

            with columna_resultados:

                st.markdown("#### Resultado del análisis")

                tarjeta(
                    "Categoría identificada",
                    categoria,
                    "Clasificación más probable",
                    destacada=True
                )

                tarjeta(
                    "Confianza del modelo",
                    f"{confianza:.2%}",
                    "Probabilidad asignada a la categoría"
                )

                st.progress(confianza)

                if confianza < UMBRAL_CONFIANZA:
                    st.warning(
                        "La confianza es baja. Se recomienda revisar "
                        "manualmente el documento antes de clasificarlo."
                    )
                else:
                    st.info(
                        "La clasificación está disponible. "
                        "Se recomienda verificarla antes de utilizarla."
                    )

                st.caption(
                    "La confianza expresa la probabilidad estimada "
                    "por el modelo, no una garantía de acierto."
                )

            st.markdown(
                '<div class="section-label">02 / PROBABILIDADES</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-heading">Distribución de predicciones</div>',
                unsafe_allow_html=True
            )

            st.altair_chart(
                grafico_probabilidades(probabilidades, categoria),
                use_container_width=True
            )

            with st.expander("Ver probabilidades de las 16 categorías"):

                tabla = pd.DataFrame({
                    "Categoría": CLASES,
                    "Probabilidad": probabilidades
                })

                tabla = tabla.sort_values(
                    "Probabilidad",
                    ascending=False
                )

                tabla["Probabilidad"] = tabla["Probabilidad"].map(
                    lambda valor: f"{valor:.2%}"
                )

                st.dataframe(
                    tabla,
                    use_container_width=True
                )

        except Exception as error:
            st.error(
                "No fue posible procesar el documento. "
                f"Detalle técnico: {error}"
            )

    else:
        st.markdown(
            """
            <div class="info-panel">
                <strong>¿Cómo funciona?</strong>
                <p>
                    1. Sube una imagen en formato PNG, JPG o TIFF.<br>
                    2. EfficientNet-B0 analiza las características visuales.<br>
                    3. Consulta la categoría, confianza y probabilidades.
                </p>
                <span style="color:#87A9C0">
                    No se requiere entrenar nuevamente el modelo.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# 8. RENDIMIENTO DEL MODELO
# =============================================================================

with tab_resultados:

    st.markdown(
        '<div class="section-label">03 / MODEL PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Indicadores de rendimiento</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Resultados obtenidos sobre las 39.999 imágenes válidas "
        "del conjunto de prueba RVL-CDIP."
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        tarjeta("Accuracy", "70,93%", "Clasificaciones correctas")

    with m2:
        tarjeta("Precision", "71,04%", "Promedio macro")

    with m3:
        tarjeta("Recall", "70,85%", "Promedio macro")

    with m4:
        tarjeta("F1-score", "70,66%", "Promedio macro")

    st.markdown(
        '<div class="section-label">04 / COMPARACIÓN</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">EfficientNet-B0 vs. ResNet50</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        RESULTADOS,
        use_container_width=True
    )

    st.markdown(
        """
        <div class="info-panel">
            <strong>¿Por qué elegimos EfficientNet-B0?</strong>
            <p>
                Ambos modelos obtuvieron resultados similares, pero
                EfficientNet-B0 alcanzó mejores métricas generales y
                necesitó menos tiempo de entrenamiento y evaluación.
                Por eso fue seleccionado para la aplicación.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Limitaciones conocidas")

    st.warning(
        "El sistema todavía presenta dificultades con documentos "
        "visualmente similares, especialmente los informes científicos. "
        "Las clasificaciones no deben utilizarse como decisiones "
        "automáticas definitivas."
    )

    st.caption(
        "Entrenamiento: muestra estratificada del 40% "
        "(128.000 imágenes de entrenamiento y 16.000 de validación). "
        "Evaluación: 39.999 imágenes válidas de prueba."
    )


# =============================================================================
# 9. ASISTENTE DE IA GENERATIVA
# =============================================================================

with tab_ia:

    st.markdown(
        '<div class="section-label">05 / GENERATIVE INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Asistente documental</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-panel">
            Esta sección utiliza Gemini para generar recomendaciones de
            organización y revisión documental a partir de las
            clasificaciones realizadas por EfficientNet-B0.
            Las sugerencias sirven como apoyo y no reemplazan
            la supervisión humana.
        </div>
        """,
        unsafe_allow_html=True
    )

    if categoria_actual is not None:

        st.metric(
            "Última categoría identificada",
            categoria_actual
        )

        st.metric(
            "Confianza de la clasificación",
            f"{confianza_actual:.2%}"
        )

        st.write(
            "Genera recomendaciones de organización y revisión "
            "a partir de la clasificación obtenida."
        )

        if st.button("Generar recomendaciones con Gemini"):

            try:
                with st.spinner("Generando recomendaciones..."):
                    recomendacion = generar_recomendacion_gemini(
                        categoria_actual,
                        confianza_actual
                    )

                st.markdown("#### Recomendaciones de Gemini")
                st.markdown(recomendacion)

                st.caption(
                    "Recomendaciones generadas mediante IA. "
                    "No reemplazan la revisión humana."
                )

            except requests.exceptions.HTTPError as error:
                codigo = (
                    error.response.status_code
                    if error.response is not None
                    else "desconocido"
                )

                st.error(
                    f"La API de Gemini devolvió un error HTTP {codigo}."
                )

            except Exception:
                st.error(
                    "No fue posible generar las recomendaciones. "
                    "Revisa la conexión y la configuración de Gemini."
                )

    else:

        st.write(
            "Primero carga un documento en la pestaña "
            "'Clasificar documento' para obtener una predicción."
        )

# =============================================================================
# 10. PIE DE PÁGINA
# =============================================================================

st.markdown(
    """
    <div class="footer">
        DOCFLOW AI · Clasificación documental con EfficientNet-B0 ·
        Proyecto académico de Inteligencia Artificial Generativa
    </div>
    """,
    unsafe_allow_html=True
)
