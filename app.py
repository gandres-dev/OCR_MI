import streamlit as st
import pdfplumber
import pandas as pd
from transformers import pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile


# ---------------------------------
import re
import unicodedata
 
# Función para normalizar texto
def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode("utf-8")
    texto = re.sub(r'\s+', ' ', texto)  # reemplazar múltiples espacios por uno
    return texto
 
# Buscar patrón en texto general
def buscar_patron(texto, patron):
    match = re.search(patron, texto, re.IGNORECASE)
    return match.group(1).strip() if match else "No encontrado"
#-------------------------------

# Cargar modelos multilingües
@st.cache_resource
def cargar_modelos():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    tqa_pipeline = pipeline("table-question-answering", model="google/tapas-large-finetuned-wtq")  # Requiere preguntas en inglés
    qa_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    mt_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    return embeddings, tqa_pipeline, qa_model, tokenizer, mt_model

from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(page_title="Resumen de expedientes", layout="wide")
st.title("🤖 Resumen de expedientes")
st.header("Dir. Metodologías y Modelos de Riesgos")

# Inicializar historial
if "historial" not in st.session_state:
    st.session_state.historial = []

uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

if uploaded_file:
    embeddings, tqa_pipeline, qa_model, tokenizer, mt_model = cargar_modelos()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        ruta_pdf = tmp.name

    all_text = ""
    tablas = []

    with pdfplumber.open(ruta_pdf) as pdf:
        for page in pdf.pages:
            texto = page.extract_text()
            if texto:
                all_text += texto + "\n"
            for tabla in page.extract_tables():
                df = pd.DataFrame(tabla[1:], columns=tabla[0])
                tablas.append(df)

    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(all_text)
    db = FAISS.from_texts(chunks, embedding=embeddings)

    def traducir_pregunta_es_en(pregunta):
        tokens = tokenizer(pregunta, return_tensors="pt", padding=True)
        traduccion = mt_model.generate(**tokens)
        return tokenizer.decode(traduccion[0], skip_special_tokens=True)

    def responder_tablas(pregunta):
        pregunta_en = traducir_pregunta_es_en(pregunta)
        for df in tablas:
            try:
                resultado = tqa_pipeline(table=df, query=pregunta_en)
                if float(resultado["score"]) > 0.5:
                    return f"(Desde tabla)\n{resultado['answer']}"
            except:
                continue
        return "No se encontró respuesta en las tablas."

    def responder_texto(pregunta):
        docs = db.similarity_search(pregunta, k=3)
        contexto = "\n".join([doc.page_content for doc in docs])
        prompt = f"""Responde en español la siguiente pregunta usando el contexto dado:

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta:"""
        respuesta = qa_model(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]
        return f"(Desde texto)\n{respuesta.split('Respuesta:')[-1].strip()}"

    st.subheader("1) Metodología con Expresiones Regulares")
    texto_normalizado = normalizar(all_text)
    # Búsqueda en texto general
    inicio_operaciones = buscar_patron(texto_normalizado, r"inicia(?:.*?en)?\s*operaciones.*?(\d{4})")
    giro_comercial = buscar_patron(texto_normalizado, r"giro comercial\s*:?(.+?)(?:\.|\n)")
    secretario = buscar_patron(texto_normalizado, r"secretario(?:a)?(?:\s*:?|\s+de\s+actas)?\s*([a-záéíóúñ\s]{5,50})")
    tesorero = buscar_patron(texto_normalizado, r"tesorero(?:a)?\s*:? ([a-záéíóúñ\s]{5,50})")

    st.markdown("""
    - Expresiones utilizadas:         
    -  `inicia(?:.*?en)?\s*operaciones.*?(\d{4})`
    -  `giro comercial\s*:?(.+?)(?:\.|\n)`
    -  `secretario(?:a)?(?:\s*:?|\s+de\s+actas)?\s*([a-záéíóúñ\s]{5,50})`
    -  `tesorero(?:a)?\s*:? ([a-záéíóúñ\s]{5,50})`        
    """)


    st.write("Inicio de operaciones:", inicio_operaciones)
    st.write("Giro comercial:", giro_comercial)
    st.write("Secretario:", secretario)
    st.write("Tesorero:", tesorero)


    st.subheader("2) Metodología con Modelos de inteligencia artificial")

    st.markdown("""
    - Modelos utilizados:         
      - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: Crea el espacio de embeddings para nuestro texto.
      - `mistralai/Mistral-7B-Instruct-v0.1`. Modelo para realizar preguntas y obtener respuesta con base a nuestro texto.
      - `google/tapas-large-finetuned-wtq`: Modelo para realizar preguntas en tablas y obtener respuesta. (Requiere preguntas en inglés). 
    """)

    with st.form("form_pregunta"):
        pregunta = st.text_input("Escribe tu pregunta sobre el PDF:")
        submit = st.form_submit_button("Preguntar")

    if submit and pregunta:
        with st.spinner("Analizando el PDF y buscando la respuesta..."):
            respuesta = responder_tablas(pregunta)
            if "No se encontró" in respuesta:
                respuesta = responder_texto(pregunta)
            st.session_state.historial.append((pregunta, respuesta))

    if st.session_state.historial:
        st.markdown("## 📝 Historial de Preguntas")
        for i, (preg, resp) in enumerate(reversed(st.session_state.historial), 1):
            st.markdown(f"**{i}. Pregunta:** {preg}")
            st.markdown(f"**Respuesta:** {resp}")
            st.markdown("---")
