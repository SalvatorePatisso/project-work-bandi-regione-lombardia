# main.py
from crewai import Crew, Process
from agents.reader_agent import ReaderAgent
from tasks.reader_tasks import ReaderTasks
import os
import pathlib
from dotenv import load_dotenv

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
    
    # Cerca file FAISS
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

def main():
    print("🚀 SISTEMA RAG + CrewAI - Analisi Intelligente Bandi Lombardia")
    print("=" * 70)
    print()
    
    # Carica variabili d'ambiente
    load_dotenv()
    
    # Validazione prerequisiti
    print("=== VALIDAZIONE SISTEMA ===")
    
    # 1. Verifica variabili d'ambiente
    if not validate_environment():
        return None
    print("✅ Variabili d'ambiente configurate correttamente")
    
    # 2. Verifica vector store
    vector_store_valid, db_folder = validate_vector_store()
    if not vector_store_valid:
        return None
    print(f"✅ Vector store trovato in: {db_folder}")
    
    # 3. Mostra configurazione
    print("\n=== CONFIGURAZIONE AZURE ===")
    print(f"Endpoint: {os.getenv('AZURE_API_BASE')}")
    print(f"LLM Model: {os.getenv('AZURE_LLM_MODEL')}")
    print(f"Embedding Model: {os.getenv('AZURE_EMBEDDING_MODEL')}")
    print(f"API Version: {os.getenv('AZURE_API_VERSION')}")
    print()
    
    # Input dell'utente
    business_idea = get_business_idea()
    print(f"\n📋 Idea di business ricevuta ({len(business_idea)} caratteri)")
    print(f"Anteprima: {business_idea[:150]}...")
    print()
    
    try:
        # Inizializzazione agente con RAG
        print("=== INIZIALIZZAZIONE SISTEMA ===")
        reader_agent_instance = ReaderAgent()
        
        # Test connessione LLM prima di procedere
        if not reader_agent_instance.test_llm_connection():
            print("❌ LLM non funziona, impossibile procedere")
            return None
            
        print("✅ Agente Reader inizializzato")
        print("✅ Sistema RAG connesso")
        print("✅ Vector store caricato")
        print("✅ LLM testato e funzionante")
        print()
        
        # Ricerca del documento più rilevante tramite RAG
        print("=== RICERCA DOCUMENTO OTTIMALE ===")
        print("🔍 Analizzando il database vettoriale per trovare il bando più adatto...")
        
        # Recupera documento + metadata (incluso nome file)
        document_context, metadata = reader_agent_instance.get_most_relevant_document(business_idea)
        
        # Estrae nome file dai metadata
        filename = reader_agent_instance.extract_filename_from_metadata(metadata)
        
        # Verifica che il documento sia stato recuperato correttamente
        if "Errore" in document_context or "Nessun documento" in document_context:
            print(f"❌ Problema nel recupero documento: {document_context}")
            return None
        
        print("✅ Documento più rilevante identificato!")
        print(f"📄 Nome file: {filename}")
        print(f"📄 Lunghezza documento: {len(document_context)} caratteri")
        print(f"📄 Anteprima documento:")
        print("-" * 50)
        print(document_context[:400] + "..." if len(document_context) > 400 else document_context)
        print("-" * 50)
        print()
        
        # Creazione crew CrewAI
        print("=== CONFIGURAZIONE ANALISI CREWAI ===")
        reader_agent = reader_agent_instance.create_agent()
        
        # Creazione task con il documento specifico trovato dal RAG
        analysis_task = ReaderTasks.create_iterative_document_analysis_task(
            agent=reader_agent,
            business_idea=business_idea,
            document_context=document_context,  # ← Il documento RAW trovato dal RAG
            filename=filename  # ← Nome file estratto dai metadata
        )
        
        # Configurazione crew
        crew = Crew(
            agents=[reader_agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True
        )
        
        print("✅ Crew configurata con successo")
        print("🤖 Agente pronto per l'analisi")
        print()
        
        # Esecuzione analisi
        print("=== ANALISI DOCUMENTO IN CORSO ===")
        print("⚙️  L'agente sta analizzando il documento trovato dal RAG...")
        print("⚙️  Estrazione informazioni chiave in corso...")
        print("⚙️  Calcolo allineamento business-bando...")
        print()
        
        result = crew.kickoff()
        
        # Presentazione risultati
        print("\n" + "=" * 70)
        print("🎉 ANALISI COMPLETATA - RISULTATI")
        print("=" * 70)
        print()
        print("📊 DOCUMENTO ANALIZZATO:")
        print(f"   Fonte: Database vettoriale RAG")
        print(f"   Rilevanza: Massima compatibilità con la tua idea di business")
        print(f"   Elaborazione: CrewAI + Azure OpenAI")
        print()
        print("📋 RISULTATO STRUTTURATO:")
        print("-" * 70)
        print(result)
        print("-" * 70)
        print()
        print("✅ Analisi completata con successo!")
        
        return result
        
    except ImportError as e:
        print(f"❌ ERRORE IMPORT: {e}")
        print("\n🔧 SOLUZIONE:")
        print("Installa tutte le dipendenze richieste:")
        print("pip install -r requirements.txt")
        return None
        
    except FileNotFoundError as e:
        print(f"❌ ERRORE FILE: {e}")
        print("\n🔧 SOLUZIONE:")
        print("Verifica che questi file/cartelle esistano:")
        print("- rag.py")
        print("- cartella db/ con vector store FAISS")
        print("- file .env con variabili Azure")
        return None
        
    except Exception as e:
        print(f"❌ ERRORE GENERALE: {e}")
        print(f"Tipo errore: {type(e).__name__}")
        print("\n🔧 POSSIBILI SOLUZIONI:")
        print("1. Verifica connessione ad Azure OpenAI")
        print("2. Controlla che il vector store sia valido")
        print("3. Verifica che i documenti PDF siano stati vettorizzati correttamente")
        print("4. Controlla i log di errore sopra per dettagli specifici")
        return None

if __name__ == "__main__":
    result = main()
    
    if result:
        print("\n" + "=" * 50)
        print("🚀 SISTEMA OPERATIVO E PRONTO!")
        print("Riavvia il programma per analizzare altre idee di business.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ SISTEMA NON OPERATIVO")
        print("Risolvi gli errori sopra indicati e riprova.")
        print("=" * 50)