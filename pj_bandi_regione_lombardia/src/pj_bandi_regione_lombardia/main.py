# main.py
from crewai import Crew, Process
from agents.reader_agent import ReaderAgent
from agents.extractor_agent import ExtractorAgent
from agents.writer_agent import WriterAgent
from tasks.extractor_tasks import ExtractorTasks
import os
import pathlib
from dotenv import load_dotenv
import json
import threading
import time

def validate_environment():
    """Valida che tutte le variabili d'ambiente necessarie siano configurate"""
    required_vars = [
        'AZURE_API_KEY',
        'AZURE_API_BASE', 
        'AZURE_API_VERSION',
        'AZURE_EMBEDDING_MODEL',
        'AZURE_LLM_MODEL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ ERRORE: Variabili d'ambiente mancanti:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nConfigura queste variabili nel file .env")
        return False
    
    return True

def validate_vector_store():
    """Valida che il vector store FAISS sia presente e accessibile"""
    base_dir = pathlib.Path(__file__).parent.parent
    db_folder = base_dir / "db"
    
    if not db_folder.exists():
        print(f"❌ ERRORE: Cartella vector store non trovata: {db_folder}")
        print("Crea la cartella 'db' e vettorizza i documenti PDF prima di procedere")
        return False, None
    
    faiss_files = list(db_folder.glob("*.faiss"))
    pkl_files = list(db_folder.glob("*.pkl"))
    
    if not faiss_files or not pkl_files:
        print(f"❌ ERRORE: File FAISS non completi nella cartella {db_folder}")
        print("File trovati:")
        print(f"   - File .faiss: {[f.name for f in faiss_files]}")
        print(f"   - File .pkl: {[f.name for f in pkl_files]}")
        print("\nEsegui la vettorizzazione dei documenti PDF prima di procedere")
        return False, None
    
    return True, db_folder

def get_business_idea():
    """Ottiene l'idea di business dall'utente con opzione di esempio"""
    print("=== INPUT IDEA DI BUSINESS ===")
    print("Inserisci la tua idea di business per trovare il bando più adatto.")
    print("Puoi descrivere:")
    print("- Il settore di attività")
    print("- La tipologia di azienda (startup, PMI, etc.)")
    print("- Le tecnologie o innovazioni che vuoi sviluppare")
    print("- Gli obiettivi del progetto")
    print()
    
    business_idea = input("La tua idea di business (o premi Enter per usare l'esempio agritech): ").strip()
    
    if not business_idea:
        business_idea = """
        Voglio sviluppare una piattaforma digitale per la gestione sostenibile 
        delle risorse idriche in agricoltura, utilizzando sensori IoT e AI per 
        ottimizzare l'irrigazione e ridurre gli sprechi. La mia azienda è una 
        startup innovativa nel settore agritech che vuole contribuire alla 
        sostenibilità ambientale e all'efficienza della produzione agricola.
        """
        print("🌱 Usando esempio: Startup agritech - Piattaforma IoT per gestione risorse idriche")
    
    return business_idea

def run_extractor_agent(extractor_agent_instance, reader_agent_instance, filename):
    """Esegue l'agente Extractor in background con approccio ibrido"""
    print("\n🤖 EXTRACTOR AGENT: Avvio estrazione dati strutturati con approccio ibrido...")
    
    try:
        # Crea la cartella json_description se non esiste
        json_dir = pathlib.Path(__file__).parent / "json_description"
        json_dir.mkdir(exist_ok=True)
        
        # Crea il nome del file JSON basato sul nome del bando
        json_filename = filename.replace('.pdf', '.json').replace('.PDF', '.json')
        output_file = json_dir / json_filename
        
        print(f"📁 Directory output: {json_dir}")
        print(f"📁 Directory output (PATH ASSOLUTO): {json_dir.absolute()}")
        print(f"📄 File output: {json_filename}")
        print(f"📄 File output (PATH COMPLETO): {output_file.absolute()}")
        
        # Recupera il path completo del file sorgente
        source_file = reader_agent_instance.current_metadata.get('source', '')
        
        # Ricostruisci il documento completo
        full_document = extractor_agent_instance.reconstruct_full_document(
            reader_agent_instance.rag_system, 
            source_file
        )
        
        if full_document:
            # Usa il nuovo metodo ibrido che combina RAG + documento completo
            extracted_data = extractor_agent_instance.extract_structured_info_hybrid(
                rag_system=reader_agent_instance.rag_system,
                full_document=full_document,
                filename=filename,
                source_file=source_file
            )
            
            if extracted_data:
                # Controlla se il file esiste già
                if output_file.exists():
                    print(f"⚠️ File esistente verrà sovrascritto: {output_file.name}")
                
                # Salva il JSON
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
                        f.flush()  # Forza la scrittura su disco
                        os.fsync(f.fileno())  # Assicura che sia scritto fisicamente
                except Exception as e:
                    print(f"\n❌ EXTRACTOR AGENT: Errore durante il salvataggio: {e}")
                    return
                
                # Verifica che il file sia stato effettivamente scritto
                if output_file.exists():
                    file_size = output_file.stat().st_size
                    print(f"\n✅ EXTRACTOR AGENT: Dati salvati in {output_file}")
                    print(f"📍 PATH COMPLETO: {output_file.absolute()}")
                    print(f"📊 Dimensione file: {file_size} bytes")
                    print(f"📊 Anteprima dati estratti:")
                    print(f"   - Ente: {extracted_data.get('Ente erogatore', 'N/A')}")
                    print(f"   - Titolo: {extracted_data.get('Titolo dell\'avviso', 'N/A')[:50]}...")
                    print(f"   - Dotazione: {extracted_data.get('Dotazione finanziaria', 'N/A')}")
                    print(f"   - Beneficiari: {extracted_data.get('Beneficiari', 'N/A')[:50]}...")
                    
                    # Conta i campi compilati vs non specificati
                    filled_fields = sum(1 for v in extracted_data.values() if v != "Non specificato")
                    total_fields = len(extracted_data)
                    print(f"   - Completezza: {filled_fields}/{total_fields} campi compilati")
                else:
                    print(f"\n❌ EXTRACTOR AGENT: Errore - il file non è stato salvato!")
            else:
                print("\n❌ EXTRACTOR AGENT: Errore nell'estrazione dati")
        else:
            print("\n❌ EXTRACTOR AGENT: Impossibile ricostruire il documento completo")
            
    except Exception as e:
        print(f"\n❌ EXTRACTOR AGENT: Errore: {e}")

def run_writer_agent(writer_agent_instance, json_dir: pathlib.Path):
    """Esegue il WriterAgent per creare il file Excel"""
    print("\n📝 WRITER AGENT: Avvio creazione report Excel...")
    
    try:
        # Crea il file Excel
        excel_path = writer_agent_instance.create_excel_file(json_dir)
        
        if excel_path:
            # Valida il file creato
            if writer_agent_instance.validate_excel_output(excel_path):
                print("\n✅ WRITER AGENT: Report Excel completato con successo!")
            else:
                print("\n❌ WRITER AGENT: Il file Excel creato non è valido")
        else:
            print("\n❌ WRITER AGENT: Impossibile creare il file Excel")
            
    except Exception as e:
        print(f"\n❌ WRITER AGENT: Errore: {e}")
        import traceback
        traceback.print_exc()

def interactive_chat_mode(reader_agent_instance):
    """Modalità chat interattiva con l'utente"""
    print("\n" + "="*70)
    print("💬 MODALITÀ CHAT INTERATTIVA")
    print("="*70)
    print(f"📄 Documento caricato: {reader_agent_instance.current_filename}")
    print("Puoi fare domande sul bando. Digita 'exit' per uscire o 'reset' per azzerare la conversazione.")
    print("-"*70)
    
    while True:
        user_input = input("\n🤔 La tua domanda: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Uscita dalla modalità chat.")
            break
        elif user_input.lower() == 'reset':
            reader_agent_instance.reset_conversation()
            print("🔄 Conversazione resettata. Puoi ricominciare con nuove domande.")
            continue
        elif not user_input:
            print("⚠️ Inserisci una domanda valida.")
            continue
        
        print("\n⏳ Elaborazione risposta...")
        response = reader_agent_instance.chat_about_document(user_input)
        
        print("\n📢 RISPOSTA:")
        print("-"*70)
        print(response)
        print("-"*70)

def main():
    print("🚀 SISTEMA RAG + CrewAI - Analisi Intelligente Bandi Lombardia")
    print("=" * 70)
    print()
    
    # Carica variabili d'ambiente
    load_dotenv()
    
    # Validazione prerequisiti
    print("=== VALIDAZIONE SISTEMA ===")
    
    if not validate_environment():
        return None
    print("✅ Variabili d'ambiente configurate correttamente")
    
    vector_store_valid, db_folder = validate_vector_store()
    if not vector_store_valid:
        return None
    print(f"✅ Vector store trovato in: {db_folder}")
    
    print("\n=== CONFIGURAZIONE AZURE ===")
    print(f"Endpoint: {os.getenv('AZURE_API_BASE')}")
    print(f"LLM Model: {os.getenv('AZURE_LLM_MODEL')}")
    print(f"Embedding Model: {os.getenv('AZURE_EMBEDDING_MODEL')}")
    print(f"API Version: {os.getenv('AZURE_API_VERSION')}")
    print()
    
    # Input dell'utente
    business_idea = get_business_idea()
    print(f"\n📋 Idea di business ricevuta ({len(business_idea)} caratteri)")
    print()
    
    try:
        # Inizializzazione agenti
        print("=== INIZIALIZZAZIONE SISTEMA ===")
        reader_agent_instance = ReaderAgent()
        extractor_agent_instance = ExtractorAgent()
        writer_agent_instance = WriterAgent()
        
        # Test connessione LLM
        if not reader_agent_instance.test_llm_connection():
            print("❌ LLM non funziona, impossibile procedere")
            return None
            
        print("✅ Reader Agent inizializzato")
        print("✅ Extractor Agent inizializzato")
        print("✅ Writer Agent inizializzato")
        print("✅ Sistema RAG connesso")
        print()
        
        # Ricerca del documento più rilevante
        print("=== RICERCA DOCUMENTO OTTIMALE ===")
        print("🔍 Analizzando il database vettoriale per trovare il bando più adatto...")
        
        document_context, metadata = reader_agent_instance.get_most_relevant_document(business_idea)
        filename = reader_agent_instance.extract_filename_from_metadata(metadata)
        
        if "Errore" in document_context or "Nessun documento" in document_context:
            print(f"❌ Problema nel recupero documento: {document_context}")
            return None
        
        print("✅ Documento più rilevante identificato!")
        print(f"📄 Nome file: {filename}")
        print(f"📄 Lunghezza documento: {len(document_context)} caratteri")
        print()
        
        # Avvia l'Extractor Agent in un thread separato
        extractor_thread = threading.Thread(
            target=run_extractor_agent,
            args=(extractor_agent_instance, reader_agent_instance, filename)
        )
        extractor_thread.daemon = True
        extractor_thread.start()
        
        # Mostra info iniziali sul documento
        print("=== INFORMAZIONI INIZIALI SUL BANDO ===")
        initial_question = f"Fornisci un riassunto generale di questo bando, evidenziando come potrebbe essere rilevante per questa idea di business: {business_idea[:200]}..."
        initial_response = reader_agent_instance.chat_about_document(initial_question)
        print(initial_response)
        
        # Avvia la modalità chat interattiva
        interactive_chat_mode(reader_agent_instance)
        
        # Attendi che l'extractor finisca (con timeout)
        print("\n⏳ Attendo completamento estrazione dati...")
        extractor_thread.join(timeout=60)  # Aumentato timeout a 60 secondi
        
        if extractor_thread.is_alive():
            print("⚠️ L'estrazione dati sta ancora procedendo in background")
            print("⏳ Attendo ancora 30 secondi...")
            extractor_thread.join(timeout=30)
            
            if extractor_thread.is_alive():
                print("❌ Timeout: l'estrazione sta impiegando troppo tempo")
        else:
            print("✅ Estrazione dati completata!")
            
            # Mostra il JSON salvato se esiste
            json_dir = pathlib.Path(__file__).parent / "json_description"
            json_filename = filename.replace('.pdf', '.json').replace('.PDF', '.json')
            json_path = json_dir / json_filename
            
            # Aggiungi un piccolo delay per assicurarsi che il file sia stato scritto
            import time
            time.sleep(0.5)
            
            if json_path.exists():
                print(f"\n📊 DATI STRUTTURATI ESTRATTI (salvati in {json_path}):")
                print(f"📍 PATH ASSOLUTO: {json_path.absolute()}")
                print(f"📊 Dimensione file: {json_path.stat().st_size} bytes")
                print("-"*70)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                print("-"*70)
            else:
                print(f"\n⚠️ File JSON non trovato in: {json_path.absolute()}")
                print("Possibili cause:")
                print("1. L'estrazione è ancora in corso")
                print("2. Si è verificato un errore durante il salvataggio")
                print("3. Il percorso del file potrebbe essere diverso")
        
        # Esegui il WriterAgent dopo che l'ExtractorAgent ha finito
        print("\n" + "="*70)
        print("🚀 AVVIO WRITER AGENT")
        print("="*70)
        
        json_dir = pathlib.Path(__file__).parent / "json_description"
        
        # Verifica che ci siano file JSON da processare
        json_files = list(json_dir.glob("*.json"))
        if json_files:
            print(f"📊 Trovati {len(json_files)} file JSON da consolidare in Excel")
            
            # Lancia il WriterAgent in un thread separato
            writer_thread = threading.Thread(
                target=run_writer_agent,
                args=(writer_agent_instance, json_dir)
            )
            writer_thread.daemon = True
            writer_thread.start()
            
            # Attendi il completamento con timeout
            print("⏳ Attendo completamento creazione Excel...")
            writer_thread.join(timeout=30)
            
            if writer_thread.is_alive():
                print("⚠️ La creazione del report Excel sta ancora procedendo...")
            else:
                print("✅ Processo WriterAgent completato!")
        else:
            print("⚠️ Nessun file JSON trovato nella directory json_description")
            print("   Il WriterAgent non verrà eseguito")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE GENERALE: {e}")
        print(f"Tipo errore: {type(e).__name__}")
        return None

if __name__ == "__main__":
    result = main()
    
    if result:
        print("\n" + "=" * 50)
        print("✅ SESSIONE COMPLETATA")
        print(f"I dati estratti sono salvati in: json_description/")
        print(f"Il report Excel è salvato in: excel_output/")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ SISTEMA NON OPERATIVO")
        print("Risolvi gli errori sopra indicati e riprova.")
        print("=" * 50)