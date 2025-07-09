# tasks/reader_tasks.py
from crewai import Task
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class BusinessIdea(BaseModel):
    """Modello per l'idea di business dell'utente"""
    description: str = Field(..., description="Descrizione dell'idea di business")
    sector: str = Field(..., description="Settore di riferimento")
    target_funding: float = Field(None, description="Importo finanziamento richiesto")
    company_size: str = Field(None, description="Dimensione azienda (startup, PMI, grande)")
    geographic_scope: str = Field(None, description="Ambito geografico")

class BandoSummaryNew(BaseModel):
    """Modello per le informazioni essenziali del bando con struttura aggiornata"""
    ente_erogatore: str = Field(..., description="Ente che eroga il bando")
    titolo_avviso: str = Field(..., description="Titolo ufficiale dell'avviso") 
    descrizione_aggiuntiva: str = Field(..., description="Descrizione sintetica del bando")
    beneficiari: str = Field(..., description="Soggetti che possono beneficiare del bando")
    apertura: str = Field(..., description="Data di apertura del bando in formato DD/MM/YYYY")
    chiusura: str = Field(..., description="Data di chiusura del bando in formato DD/MM/YYYY")
    dotazione_finanziaria: str = Field(..., description="Dotazione finanziaria totale del bando")
    contributo: str = Field(..., description="Tipo e importo del contributo erogabile")
    parole_chiave: str = Field(..., description="Parole chiave principali del bando")
    aperto: str = Field(..., description="Stato del bando: 'si' se aperto, 'no' se chiuso")
    nome_file: str = Field(..., description="Nome del file sorgente del documento")

class ReaderTasks:
    """Classe per la gestione dei task dell'agente Reader con approccio ibrido"""
    
    @staticmethod
    def create_hybrid_document_analysis_task(agent, business_idea: str, document_context: str, filename: str) -> Task:
        """
        Task principale che usa ricerca ibrida (similarity + keywords) per estrarre informazioni specifiche
        
        Args:
            agent: L'agente Reader configurato con tool ibridi
            business_idea: L'idea di business dell'utente
            document_context: Il contenuto del documento trovato dal RAG
            filename: Nome del file del documento
        
        Returns:
            Task configurato per l'analisi ibrida
        """
        
        return Task(
            description=f"""
            Analizza il seguente documento di bando usando ricerca ibrida (similarity search + keyword matching) 
            per estrarre ogni informazione specifica con alta precisione.
            
            IDEA DI BUSINESS DELL'UTENTE:
            {business_idea}
            
            DOCUMENTO BANDO (estratto dal RAG):
            {document_context[:800] if len(document_context) > 800 else document_context}
            
            NOME FILE SORGENTE:
            {filename}
            
            PROCESSO DI ESTRAZIONE IBRIDA:
            
            1. USA HybridDocumentAnalysisTool con field_name="ente_erogatore" 
               per identificare l'ente che eroga il bando (Regione Lombardia, Camera di Commercio, etc.)
            
            2. USA HybridDocumentAnalysisTool con field_name="titolo_avviso" 
               per estrarre il titolo ufficiale del bando/avviso
            
            3. USA HybridDocumentAnalysisTool con field_name="apertura" 
               per trovare la data di apertura/pubblicazione del bando
            
            4. USA HybridDocumentAnalysisTool con field_name="chiusura" 
               per trovare la data di scadenza/chiusura del bando
            
            5. USA HybridDocumentAnalysisTool con field_name="dotazione_finanziaria" 
               per identificare il budget totale disponibile
            
            6. USA HybridDocumentAnalysisTool con field_name="contributo" 
               per determinare il tipo e l'importo del contributo erogabile
            
            7. USA HybridDocumentAnalysisTool con field_name="beneficiari" 
               per identificare chi può partecipare al bando
            
            8. USA BusinessAlignmentTool per valutare l'allineamento tra l'idea di business e il bando
            
            9. USA JsonBuilderTool per costruire il JSON finale strutturato includendo:
               - Tutti i campi estratti dai passi precedenti
               - "Descrizione aggiuntiva": sintesi del bando basata sui dati estratti
               - "Parole chiave": 5-7 parole chiave principali identificate nel documento
               - "Aperto": "si" o "no" basandoti sulle date estratte
               - "Nome file": "{filename}"
            
            METODOLOGIA RICERCA IBRIDA:
            - Similarity search semantica per trovare sezioni rilevanti
            - Keyword matching per precisione su informazioni specifiche  
            - Context expansion automatica intorno alle keywords
            - Ranking intelligente che combina rilevanza e presenza keywords
            - Filtro sul documento specifico per maggiore precisione
            
            IMPORTANTE:
            - Utilizza SEMPRE i tool nell'ordine specificato
            - Ogni tool HybridDocumentAnalysisTool opera sul documento specifico trovato dal RAG
            - La ricerca ibrida garantisce maggiore precisione rispetto ai soli prompt LLM
            - Il JSON finale deve rispettare ESATTAMENTE la struttura con le chiavi richieste
            
            OBIETTIVO:
            Produrre un JSON strutturato ad alta precisione con tutte le informazioni del bando
            estratte tramite metodologia ibrida avanzata.
            """,
            agent=agent,
            expected_output="""JSON strutturato con alta precisione contenente tutte le chiavi richieste:
            "Ente erogatore", "Titolo dell'avviso", "Descrizione aggiuntiva", "Beneficiari", 
            "Apertura", "Chiusura", "Dotazione finanziaria", "Contributo", "Parole chiave", 
            "Aperto", "Nome file". Ogni campo estratto tramite ricerca ibrida dal documento specifico.""",
            output_json=BandoSummaryNew,
            async_execution=False
        )
    
    @staticmethod
    def create_document_analysis_task(agent, business_idea: str, document_context: str) -> Task:
        """
        Task legacy per compatibilità con chiamate esistenti
        Reindirizza automaticamente al nuovo processo ibrido
        
        Args:
            agent: L'agente Reader
            business_idea: L'idea di business dell'utente  
            document_context: Il contenuto del documento
        
        Returns:
            Task configurato per l'analisi ibrida
        """
        
        return ReaderTasks.create_hybrid_document_analysis_task(
            agent=agent,
            business_idea=business_idea, 
            document_context=document_context,
            filename="Non specificato"  # Fallback se filename non fornito
        )
    
    @staticmethod
    def create_summary_generation_task(agent, analysis_result: str) -> Task:
        """
        Task per generazione riassunto (mantenuto per compatibilità)
        Ora reindirizza al processo ibrido principale
        
        Args:
            agent: L'agente Reader
            analysis_result: Risultato dell'analisi precedente
        
        Returns:
            Task configurato per generazione riassunto
        """
        
        return Task(
            description=f"""
            Genera un riassunto strutturato basandoti sui risultati dell'analisi ibrida:
            
            RISULTATO ANALISI IBRIDA:
            {analysis_result}
            
            Utilizza JsonBuilderTool per strutturare le informazioni nel formato richiesto.
            Assicurati che il JSON contenga tutte le chiavi necessarie con la formattazione corretta.
            """,
            agent=agent,
            expected_output="JSON strutturato con le informazioni del bando",
            output_json=BandoSummaryNew,
            async_execution=False
        )
    
    @staticmethod
    def create_multi_document_analysis_task(agent, business_idea: str, documents_list: List[Dict]) -> Task:
        """
        Task per analisi comparativa di più documenti (funzionalità futura)
        
        Args:
            agent: L'agente Reader
            business_idea: L'idea di business dell'utente
            documents_list: Lista di documenti da analizzare
        
        Returns:
            Task per analisi multi-documento
        """
        
        return Task(
            description=f"""
            Analizza e confronta i seguenti documenti di bando per l'idea di business:
            
            IDEA DI BUSINESS:
            {business_idea}
            
            DOCUMENTI DA ANALIZZARE:
            {len(documents_list)} documenti forniti
            
            Per ogni documento:
            1. Applica il processo di ricerca ibrida
            2. Estrai le informazioni strutturate
            3. Calcola l'allineamento con l'idea di business
            
            Fornisci un ranking dei bandi più promettenti con raccomandazioni.
            """,
            agent=agent,
            expected_output="Analisi comparativa con ranking dei migliori bandi",
            async_execution=False
        )