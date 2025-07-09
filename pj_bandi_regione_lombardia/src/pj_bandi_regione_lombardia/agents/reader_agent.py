# agents/reader_agent.py
from crewai import Agent
from crewai.llm import LLM
from rag import RagSystem
import os
import pathlib
from typing import List, Dict

class ReaderAgent:
    def __init__(self):
        # Configurazione Azure OpenAI con crewai.llm.LLM
        print("Configurazione LLM CrewAI...")
        self.llm = LLM(
            model=f"azure/{os.getenv('AZURE_LLM_MODEL')}",
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.7,
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
        
        # Stato della conversazione
        self.conversation_history = []
        self.current_document = None
        self.current_metadata = None
        self.current_filename = None
        
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
            filename = os.path.basename(source_path)
            return filename
        return "Non specificato"
    
    def get_most_relevant_document(self, business_idea: str) -> tuple:
        """Recupera il documento RAW più rilevante dal vector store"""
        try:
            if self.rag_system.vector_store is None:
                return "Errore: Vector store non inizializzato.", {}
            
            print(f"🔍 Cercando documento più rilevante per: {business_idea[:100]}...")
            
            documents = self.rag_system.vector_store.similarity_search(
                query=business_idea, 
                k=1
            )
            
            if documents and len(documents) > 0:
                most_relevant_doc = documents[0]
                document_content = most_relevant_doc.page_content
                metadata = getattr(most_relevant_doc, 'metadata', {})
                
                # Salva il documento corrente per la chat
                self.current_document = document_content
                self.current_metadata = metadata
                self.current_filename = self.extract_filename_from_metadata(metadata)
                
                print(f"✅ Documento trovato! Lunghezza: {len(document_content)} caratteri")
                if metadata:
                    print(f"📄 Metadata: {metadata}")
                
                return document_content, metadata
            else:
                print("❌ Nessun documento rilevante trovato nel database.")
                return "Nessun documento rilevante trovato nel database vettoriale.", {}
                
        except Exception as e:
            error_msg = f"Errore nel recupero documento RAG: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, {}
    
    def chat_about_document(self, user_question: str) -> str:
        """Gestisce una conversazione continua sul documento corrente"""
        if not self.current_document:
            return "Nessun documento caricato. Cerca prima un documento con un'idea di business."
        
        # Aggiungi la domanda alla storia
        self.conversation_history.append({"role": "user", "content": user_question})
        
        # Costruisci il contesto della conversazione
        conversation_context = f"""
        Stai analizzando il seguente documento di bando:
        File: {self.current_filename}
        
        ESTRATTO DEL DOCUMENTO:
        {self.current_document[:2000]}...
        
        STORICO CONVERSAZIONE:
        """
        
        # Aggiungi gli ultimi 5 scambi
        for exchange in self.conversation_history[-10:]:
            conversation_context += f"\n{exchange['role'].upper()}: {exchange['content']}"
        
        # Prompt per rispondere
        prompt = f"""
        {conversation_context}
        
        Rispondi alla domanda dell'utente basandoti sul documento del bando.
        Sii preciso e fai riferimento a sezioni specifiche quando possibile.
        Se l'informazione richiesta non è presente nel documento, dillo chiaramente.
        """
        
        try:
            # Usa il RAG system per cercare informazioni specifiche nel documento
            rag_response = self.rag_system.generate(user_question, k=3)
            
            # Combina la risposta RAG con il contesto della conversazione
            final_prompt = f"""
            Basandoti su queste informazioni dal documento:
            {rag_response.content}
            
            E considerando il contesto della conversazione:
            {conversation_context}
            
            Fornisci una risposta completa e contestualizzata alla domanda dell'utente.
            """
            
            response = self.llm.call(final_prompt)
            
            # Salva la risposta nella storia
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            error_msg = f"Errore durante la chat: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def reset_conversation(self):
        """Resetta la conversazione mantenendo il documento"""
        self.conversation_history = []
        print("✅ Conversazione resettata")
    
    def create_chat_agent(self) -> Agent:
        """Crea un agente specifico per la chat interattiva"""
        return Agent(
            role="Interactive Grant Document Assistant",
            goal="Rispondere a domande specifiche dell'utente sul documento di bando selezionato, mantenendo il contesto della conversazione",
            backstory="""
            Sei un assistente esperto nell'analisi di bandi pubblici che mantiene una conversazione
            fluida e contestualizzata con l'utente. 
            
            Le tue competenze includono:
            - Mantenere il contesto delle domande precedenti
            - Fornire risposte precise basate sul documento
            - Suggerire informazioni correlate quando pertinente
            - Guidare l'utente nella comprensione del bando
            
            Ricordi sempre le domande precedenti e costruisci le risposte in modo coerente.
            """,
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            memory=True
        )