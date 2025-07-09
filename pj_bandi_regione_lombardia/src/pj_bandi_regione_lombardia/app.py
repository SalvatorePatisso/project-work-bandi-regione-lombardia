# cd project-work-bandi-regione-lombardia\pj_bandi_regione_lombardia\src\pj_bandi_regione_lombardia
# python -m streamlit run app.py


import streamlit as st
import pathlib
import json
import os
from dotenv import load_dotenv

from agents import ReaderAgent, ExtractorAgent, WriterAgent, vectorize_pdfs

def main():
    st.set_page_config(page_title="RAG + CrewAI Bandi Lombardia", page_icon="🚀")
    st.title("🚀 Sistema RAG + CrewAI - Analisi Intelligente Bandi Lombardia")

    load_dotenv()

    # Percorsi cartelle
    base_dir = pathlib.Path(__file__).parent
    db_folder = pathlib.Path(r"C:\percorso\db")
    json_dir = pathlib.Path(r"C:\percorso\json_description")
    json_dir.mkdir(exist_ok=True)

    # Controllo presenza Vector Store
    db_ready = db_folder.exists() and any(db_folder.glob("*.faiss")) and any(db_folder.glob("*.pkl"))
    if not db_ready:
        st.warning("❌ Vector store non trovato o incompleto: 'db'. Carica PDF e vettorizza.")
        uploaded_files = st.file_uploader("Carica PDF da vettorizzare", accept_multiple_files=True, type=['pdf'])
        if uploaded_files:
            vectorize_pdfs(uploaded_files, db_folder, st)
        st.stop()
    st.success("✅ Vector store presente e pronto")

    # Controllo variabili ambiente richieste
    required_vars = [
        'AZURE_API_KEY',
        'AZURE_API_BASE',
        'AZURE_API_VERSION',
        'AZURE_EMBEDDING_MODEL',
        'AZURE_LLM_MODEL'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        st.error(f"❌ Variabili d'ambiente mancanti: {', '.join(missing_vars)}")
        st.stop()
    st.success("✅ Variabili d'ambiente caricate")

    # Input idea di business
    business_idea = st.text_area("Descrivi la tua idea di business:", height=150)
    if not business_idea.strip():
        st.info("Inserisci un'idea di business per continuare")
        st.stop()

    # Avvio analisi e ricerca bando più rilevante
    if st.button("🔍 Avvia analisi"):
        reader_agent = ReaderAgent()
        extractor_agent = ExtractorAgent()
        writer_agent = WriterAgent()

        if not reader_agent.test_llm_connection():
            st.error("❌ LLM non risponde, verifica chiavi Azure")
            st.stop()
        st.success("✅ LLM connesso")

        with st.spinner("Ricerca bando più rilevante..."):
            doc_context, metadata = reader_agent.get_most_relevant_document(business_idea)
        filename = reader_agent.extract_filename_from_metadata(metadata)

        if "Errore" in doc_context or "Nessun documento" in doc_context:
            st.error(f"❌ Errore nel recupero documento: {doc_context}")
            st.stop()

        st.success(f"✅ Documento selezionato: {filename}")
        st.text(f"Contenuto documento: {doc_context[:500]}...")

        # Salvo agenti e file in session_state
        st.session_state['filename'] = filename
        st.session_state['reader_agent'] = reader_agent
        st.session_state['extractor_agent'] = extractor_agent
        st.session_state['writer_agent'] = writer_agent
        st.session_state['json_dir'] = json_dir

        # Reset chat
        st.session_state['chat_history'] = []

    # Chat interattiva se agent già disponibili
    if 'reader_agent' in st.session_state:
        st.markdown("### 💬 Chat sul bando selezionato")

        user_input = st.text_input("Fai una domanda o digita 'reset' o 'exit'")
        if user_input:
            if user_input.lower() == "exit":
                st.session_state['chat_history'] = []
                st.info("Conversazione terminata")
            elif user_input.lower() == "reset":
                st.session_state['reader_agent'].reset_conversation()
                st.session_state['chat_history'] = []
                st.info("Conversazione resettata")
            else:
                with st.spinner("Generazione risposta..."):
                    response = st.session_state['reader_agent'].chat_about_document(user_input)
                st.session_state['chat_history'].append({"Q": user_input, "A": response})

        for chat in st.session_state['chat_history']:
            st.markdown(f"**Tu:** {chat['Q']}")
            st.markdown(f"**Risposta:** {chat['A']}")

        # Estrazione dati strutturati
        st.markdown("---")
        if st.button("📊 Estrai dati strutturati dal bando"):
            source_file = st.session_state['reader_agent'].current_metadata.get('source', '')
            full_document = st.session_state['extractor_agent'].reconstruct_full_document(
                st.session_state['reader_agent'].rag_system, source_file
            )

            if full_document:
                extracted_data = st.session_state['extractor_agent'].extract_structured_info_hybrid(
                    rag_system=st.session_state['reader_agent'].rag_system,
                    full_document=full_document,
                    filename=st.session_state['filename'],
                    source_file=source_file
                )

                if extracted_data:
                    json_path = st.session_state['json_dir'] / (st.session_state['filename'].replace(".pdf", ".json"))
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
                    st.success(f"Dati estratti salvati in {json_path.name}")
                    st.session_state['extracted_json_path'] = json_path
                    st.json(extracted_data)
                else:
                    st.error("Errore durante l'estrazione dei dati")
            else:
                st.error("Impossibile ricostruire il documento per l'estrazione")

    # Creazione report Excel se dati estratti
    if 'extracted_json_path' in st.session_state:
        st.markdown("---")
        if st.button("📥 Crea e scarica report Excel"):
            excel_path = st.session_state['writer_agent'].create_excel_file(st.session_state['json_dir'])

            if excel_path:
                excel_path = pathlib.Path(excel_path)
                if st.session_state['writer_agent'].validate_excel_output(excel_path):
                    st.success(f"✅ Report Excel creato: {excel_path.name}")
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="Scarica il report Excel",
                            data=f,
                            file_name=excel_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("Errore nella validazione del report Excel")
            else:
                st.error("Errore nella creazione del report Excel")

if __name__ == "__main__":
    main()