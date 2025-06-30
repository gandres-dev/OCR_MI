import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import torch

# Modelo multilingüe para español
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

st.set_page_config(page_title="Asistente Financiero PDF (Español)", layout="wide")
st.title("💼 Asistente Financiero PDF en Español (Gratis y Local)")
col1, col2 = st.columns([1, 2])

def extract_text_and_tables(file):
    text_chunks = []
    all_tables = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                for para in text.split("\n"):
                    if len(para.strip()) > 30:
                        text_chunks.append(para.strip())
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join(["\t".join(row) for row in table])
                text_chunks.append("[TABLA FINANCIERA]\n" + table_text)
                all_tables.append(table)
    return text_chunks, all_tables

# 📄 Carga del PDF
with col2:
    st.header("📎 Subir archivo PDF financiero")
    uploaded_file = st.file_uploader("Selecciona un archivo PDF", type="pdf")

    if uploaded_file is not None:
        st.success("✅ PDF cargado")
        with st.spinner("🧠 Analizando contenido financiero..."):
            chunks, tables = extract_text_and_tables(uploaded_file)
            corpus_embeddings = model.encode(chunks, convert_to_tensor=True)
        st.subheader("📖 Fragmentos del documento")
        for i, chunk in enumerate(chunks[:]):
            st.markdown(f"**Fragmento {i+1}:** {chunk}")

        if tables:
            st.subheader("📋 Tablas financieras")
            for i, table in enumerate(tables):
                st.write(f"Tabla {i+1}")
                st.dataframe(table)
    else:
        st.info("Por favor, sube un PDF para comenzar.")

# ❓ Preguntas sobre finanzas
with col1:
    st.header("💬 Haz una pregunta financiera")
    if uploaded_file is not None:
        question = st.text_input("Ejemplo: ¿Cuál es el flujo de efectivo neto del trimestre?")
        if question:
            with st.spinner("Buscando respuesta basada en el PDF..."):
                question_embedding = model.encode(question, convert_to_tensor=True)
                scores = util.pytorch_cos_sim(question_embedding, corpus_embeddings)[0]
                top_k = torch.topk(scores, k=3)
                st.subheader("📌 Fragmentos relevantes encontrados:")
                for idx in top_k.indices:
                    st.markdown(f"- {chunks[idx]}")
    else:
        st.warning("Primero sube un archivo PDF financiero.")
