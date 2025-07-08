# agents/reader_agent.py
from crewai import Agent
from crewai.llm import LLM
from tools.reader_tools import DocumentAnalysisTool, BusinessAlignmentTool
from rag import RagSystem
import os
import pathlib

class ReaderAgent:
    def __init__(self):
        # Configurazione Azure OpenAI con crewai.llm.LLM
        self.llm = LLM(
            model=f"azure/{os.getenv('AZURE_LLM_MODEL')}",
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.3,
            max_tokens=4000
        )
        
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
        
        # Inizializzazione tools
        self.document_analysis_tool = DocumentAnalysisTool()
        self.business_alignment_tool = BusinessAlignmentTool()
    
    def get_most_relevant_document(self, business_idea: str) -> str:
        """
        Recupera il documento RAW più rilevante dal vector store
        Questo metodo restituisce il contenuto originale del documento PDF, 
        non una risposta elaborata dal LLM.
        """
        try:
            if self.rag_system.vector_store is None:
                return "Errore: Vector store non inizializzato."
            
            print(f"Cercando documento più rilevante per: {business_idea[:100]}...")
            
            # Usa similarity_search per ottenere i documenti raw dal vector store
            # k=1 significa che vogliamo solo il documento più simile
            documents = self.rag_system.vector_store.similarity_search(
                query=business_idea, 
                k=1
            )
            
            if documents and len(documents) > 0:
                # Estrae il contenuto del documento più rilevante
                most_relevant_doc = documents[0]
                document_content = most_relevant_doc.page_content
                
                # Informazioni aggiuntive sul documento (se disponibili)
                metadata = getattr(most_relevant_doc, 'metadata', {})
                
                print(f"✅ Documento trovato! Lunghezza: {len(document_content)} caratteri")
                if metadata:
                    print(f"Metadata: {metadata}")
                
                # Restituisce solo il contenuto del documento
                return document_content
            else:
                print("❌ Nessun documento rilevante trovato nel database.")
                return "Nessun documento rilevante trovato nel database vettoriale."
                
        except Exception as e:
            error_msg = f"Errore nel recupero documento RAG: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def get_rag_elaborated_answer(self, business_idea: str) -> str:
        """
        Recupera una risposta elaborata dal sistema RAG (con LLM)
        Questo metodo usa il tuo prompt template e restituisce una risposta strutturata.
        """
        try:
            print("Generando risposta elaborata dal sistema RAG...")
            answer = self.rag_system.generate(business_idea, k=3)
            return answer.content
        except Exception as e:
            error_msg = f"Errore nella generazione risposta RAG: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def search_multiple_documents(self, business_idea: str, k: int = 3) -> list:
        """
        Recupera i top K documenti più rilevanti
        Utile per analisi comparative o quando serve più contesto.
        """
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
                results.append({
                    'rank': i + 1,
                    'content': doc.page_content,
                    'metadata': getattr(doc, 'metadata', {}),
                    'length': len(doc.page_content)
                })
            
            print(f"✅ Trovati {len(results)} documenti")
            return results
            
        except Exception as e:
            print(f"❌ Errore nella ricerca multipla: {e}")
            return []
    
    def create_agent(self) -> Agent:
        """Crea e configura l'agente Reader con CrewAI LLM"""
        
        return Agent(
            role="Expert Document Analyst for Grant Applications",
            goal="Analizzare documenti di bandi recuperati tramite sistema RAG e fornire informazioni essenziali strutturate per supportare decisioni di business",
            backstory="""
            Sei un esperto analista di bandi pubblici e finanziamenti aziendali della Regione Lombardia.
            
            Le tue competenze specialistiche includono:
            - Analisi approfondita di documenti di bandi e avvisi pubblici
            - Identificazione rapida di criteri di eleggibilità e requisiti
            - Valutazione dell'allineamento tra progetti imprenditoriali e opportunità di finanziamento  
            - Estrazione di informazioni critiche come importi, scadenze e procedure
            - Sintesi di documenti complessi in formato strutturato e actionable
            
            Utilizzi un sistema RAG avanzato che ti permette di accedere istantaneamente 
            al documento più pertinente dal database vettorizzato di bandi della Lombardia.
            
            Il tuo obiettivo è fornire analisi precise, complete e orientate all'azione 
            per aiutare imprenditori e aziende a identificare le migliori opportunità di finanziamento.
            """,
            verbose=True,
            allow_delegation=False,
            tools=[
                self.document_analysis_tool,
                self.business_alignment_tool
            ],
            llm=self.llm,
            max_iter=3,
            memory=True
        )