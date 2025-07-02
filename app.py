import streamlit as st
import pdfplumber
import pandas as pd
from transformers import pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile

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
