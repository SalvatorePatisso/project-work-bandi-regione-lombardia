# agents/reader_agent.py
from crewai import Agent
from crewai.llm import LLM
from tools.reader_tools import LLMDocumentExtractorTool, BusinessAlignmentTool, JsonBuilderTool
from rag import RagSystem
import os
import pathlib

class ReaderAgent:
    def __init__(self):
        # Configurazione Azure OpenAI con crewai.llm.LLM
        print("Configurazione LLM CrewAI...")
        self.llm = LLM(
            model=f"azure/{os.getenv('AZURE_LLM_MODEL')}",
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.1,  # Ridotta per maggiore precisione nell'estrazione
            max_tokens=4000
        )
        print(f"✅ LLM configurato con model: azure/{os.getenv('AZURE_LLM_MODEL')}")
        
        # Inizializzazione sistema RAG
        print("Inizializzazione sistema RAG...")
        self.rag_system = RagSystem(
            api_key=os.getenv("AZURE_API_KEY"),
            api_end_point=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            embedding_model=os.getenv("AZURE_EMBEDDING_MODEL"),
            llm_model=os.getenv("AZURE_LLM_MODEL")
        )
        
        # Carica vector store esistente
        base_dir = pathlib.Path(__file__).parent.parent.parent
        db_folder = base_dir / "db"
        
        if not db_folder.exists():
            raise FileNotFoundError(f"Cartella vector store non trovata: {db_folder}")
        
        print(f"Caricamento vector store da: {db_folder}")
        self.rag_system.load_vector_store(vector_store_path=str(db_folder))
        print("Vector store caricato con successo!")
        
        # Inizializzazione tool LLM-based
        self.llm_extractor_tool = LLMDocumentExtractorTool()
        self.business_alignment_tool = BusinessAlignmentTool()
        self.json_builder_tool = JsonBuilderTool()
        
        print("✅ Tool LLM-based inizializzati:")
        print("   - LLMDocumentExtractorTool (estrazione iterativa)")
        print("   - BusinessAlignmentTool (valutazione allineamento)")
        print("   - JsonBuilderTool (costruzione JSON finale)")
    
    def get_most_relevant_document(self, business_idea: str) -> tuple:
        """
        Recupera il documento RAW più rilevante dal vector store
        Restituisce tupla (document_content, metadata) per includere nome file
        """
        try:
            if self.rag_system.vector_store is None:
                return "Errore: Vector store non inizializzato.", {}
            
            print(f"🔍 Cercando documento più rilevante per: {business_idea[:100]}...")
            
            # Usa similarity_search per ottenere i documenti raw dal vector store
            documents = self.rag_system.vector_store.similarity_search(
                query=business_idea, 
                k=1
            )
            
            if documents and len(documents) > 0:
                # Estrae il contenuto del documento più rilevante
                most_relevant_doc = documents[0]
                document_content = most_relevant_doc.page_content
                
                # Informazioni aggiuntive sul documento (incluso nome file)
                metadata = getattr(most_relevant_doc, 'metadata', {})
                
                print(f"✅ Documento trovato! Lunghezza: {len(document_content)} caratteri")
                if metadata:
                    print(f"📄 Metadata: {metadata}")
                
                # Restituisce contenuto E metadata
                return document_content, metadata
            else:
                print("❌ Nessun documento rilevante trovato nel database.")
                return "Nessun documento rilevante trovato nel database vettoriale.", {}
                
        except Exception as e:
            error_msg = f"Errore nel recupero documento RAG: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, {}
    
    def test_llm_connection(self):
        """Testa la connessione LLM prima di usare l'agente"""
        try:
            print("🧪 Test connessione LLM...")
            response = self.llm.call("Rispondi solo 'OK' se funzioni.")
            print(f"✅ LLM funziona: {response}")
            return True
        except Exception as e:
            print(f"❌ LLM non funziona: {e}")
            return False
    
    def extract_filename_from_metadata(self, metadata: dict) -> str:
        """Estrae il nome del file dai metadata"""
        if 'source' in metadata:
            source_path = metadata['source']
            # Estrae solo il nome del file dal percorso completo
            filename = os.path.basename(source_path)
            return filename
        
        return "Non specificato"
    
    def test_llm_extraction(self, document_text: str, field_name: str) -> str:
        """Test singolo campo per debugging"""
        try:
            prompt = self.llm_extractor_tool._run(document_text, field_name)
            response = self.llm.call(prompt)
            print(f"✅ Test {field_name}: {response[:100]}...")
            return response
        except Exception as e:
            print(f"❌ Errore test {field_name}: {e}")
            return f"Errore: {e}"
    
    def create_agent(self) -> Agent:
        """Crea e configura l'agente Reader con tool LLM iterativi"""
        
        return Agent(
            role="Expert LLM-based Document Analyst for Grant Applications",
            goal="Analizzare documenti di bandi usando l'LLM per estrarre iterativamente ogni informazione specifica richiesta",
            backstory="""
            Sei un esperto analista di bandi pubblici specializzato nell'uso di Large Language Models 
            per l'estrazione precisa di informazioni da documenti complessi.
            
            Le tue competenze distintive includono:
            - Utilizzo dell'LLM per fare domande mirate e specifiche sui documenti
            - Estrazione iterativa di informazioni attraverso prompt strutturati
            - Analisi contestuale per identificare enti erogatori, date, importi e requisiti
            - Valutazione dell'allineamento tra progetti imprenditoriali e opportunità di finanziamento
            - Costruzione di output strutturati in formato JSON con precisione
            
            Il tuo approccio metodologico prevede:
            1. Analisi del documento attraverso domande specifiche per ogni campo
            2. Utilizzo dell'intelligenza del modello LLM per interpretare il contesto
            3. Estrazione sistematica di tutte le informazioni richieste
            4. Validazione e strutturazione finale dei dati
            
            Non usi regex o pattern matching, ma ti affidi completamente all'intelligenza 
            del modello LLM per comprendere e interpretare i contenuti dei documenti.
            """,
            verbose=True,
            allow_delegation=False,
            tools=[
                self.llm_extractor_tool,
                self.business_alignment_tool, 
                self.json_builder_tool
            ],
            llm=self.llm,
            max_iter=15,  # Aumentato per gestire processo iterativo
            memory=True
        )
    
    def get_rag_elaborated_answer(self, business_idea: str) -> str:
        """Recupera una risposta elaborata dal sistema RAG (con LLM)"""
        try:
            print("Generando risposta elaborata dal sistema RAG...")
            answer = self.rag_system.generate(business_idea, k=3)
            return answer.content
        except Exception as e:
            error_msg = f"Errore nella generazione risposta RAG: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def search_multiple_documents(self, business_idea: str, k: int = 3) -> list:
        """Recupera i top K documenti più rilevanti"""
        try:
            if self.rag_system.vector_store is None:
                return []
            
            print(f"Cercando i top {k} documenti più rilevanti...")
            
            documents = self.rag_system.vector_store.similarity_search(
                query=business_idea, 
                k=k
            )
            
            results = []
            for i, doc in enumerate(documents):
                metadata = getattr(doc, 'metadata', {})
                filename = self.extract_filename_from_metadata(metadata)
                
                results.append({
                    'rank': i + 1,
                    'content': doc.page_content,
                    'metadata': metadata,
                    'filename': filename,
                    'length': len(doc.page_content)
                })
            
            print(f"✅ Trovati {len(results)} documenti")
            return results
            
        except Exception as e:
            print(f"❌ Errore nella ricerca multipla: {e}")
            return []