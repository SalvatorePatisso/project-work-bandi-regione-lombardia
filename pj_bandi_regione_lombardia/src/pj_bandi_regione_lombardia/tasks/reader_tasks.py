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
    """Modello per le informazioni essenziali del bando"""
    ente_erogatore: str = Field(..., description="Ente che eroga il bando")
    titolo_avviso: str = Field(..., description="Titolo ufficiale dell'avviso")
    descrizione_aggiuntiva: str = Field(..., description="Descrizione sintetica del bando")
    beneficiari: str = Field(..., description="Soggetti che possono beneficiare del bando")
    apertura: str = Field(..., description="Data di apertura del bando")
    chiusura: str = Field(..., description="Data di chiusura del bando")
    dotazione_finanziaria: str = Field(..., description="Dotazione finanziaria totale del bando")
    contributo: str = Field(..., description="Tipo e importo del contributo")
    parole_chiave: str = Field(..., description="Parole chiave principali del bando")
    aperto: str = Field(..., description="Stato del bando: 'si' se aperto, 'no' se chiuso")
    nome_file: str = Field(..., description="Nome del file sorgente del documento")

class ReaderTasks:
    
    @staticmethod
    def create_iterative_document_analysis_task(agent, business_idea: str, document_context: str, filename: str) -> Task:
        """Task che usa l'LLM per estrarre iterativamente ogni campo del JSON"""
        
        return Task(
            description=f"""
            Analizza il seguente documento di bando usando l'LLM per estrarre ogni informazione specifica.
            Devi seguire un processo iterativo dove per ogni campo utilizzi l'LLM per fare una domanda mirata.
            
            IDEA DI BUSINESS DELL'UTENTE:
            {business_idea}
            
            DOCUMENTO BANDO DA ANALIZZARE:
            {document_context}
            
            NOME FILE:
            {filename}
            
            PROCESSO ITERATIVO DA SEGUIRE:
            
            1. USA LLMDocumentExtractorTool con field_name="ente_erogatore" per identificare l'ente che eroga il bando
            
            2. USA LLMDocumentExtractorTool con field_name="titolo_avviso" per estrarre il titolo ufficiale
            
            3. USA LLMDocumentExtractorTool con field_name="descrizione_aggiuntiva" per creare una descrizione sintetica
            
            4. USA LLMDocumentExtractorTool con field_name="beneficiari" per identificare chi può partecipare
            
            5. USA LLMDocumentExtractorTool con field_name="apertura" per trovare la data di apertura
            
            6. USA LLMDocumentExtractorTool con field_name="chiusura" per trovare la data di chiusura
            
            7. USA LLMDocumentExtractorTool con field_name="dotazione_finanziaria" per il budget totale
            
            8. USA LLMDocumentExtractorTool con field_name="contributo" per tipo e importo del contributo
            
            9. USA LLMDocumentExtractorTool con field_name="parole_chiave" per le parole chiave principali
            
            10. USA LLMDocumentExtractorTool con field_name="aperto" per determinare se è aperto o chiuso
            
            11. USA BusinessAlignmentTool per valutare l'allineamento con l'idea di business
            
            12. USA JsonBuilderTool per costruire il JSON finale con TUTTE le informazioni estratte, incluso:
                - Tutti i campi estratti dai passi 1-10
                - Nome file: "{filename}"
                - Informazioni sull'allineamento dal passo 11
            
            IMPORTANTE:
            - Esegui OGNI passo in sequenza
            - Per ogni campo, usa l'LLM per fare una domanda specifica sul documento
            - Non usare regex o pattern matching
            - Raccogli tutte le risposte e costruisci il JSON finale
            - Il JSON deve avere esattamente le chiavi richieste con gli spazi e maiuscole corrette
            
            OBIETTIVO FINALE:
            Produrre un JSON strutturato con tutte le informazioni del bando estratte tramite LLM.
            """,
            agent=agent,
            expected_output="""JSON strutturato con le chiavi:
            "Ente erogatore", "Titolo dell'avviso", "Descrizione aggiuntiva", "Beneficiari", 
            "Apertura", "Chiusura", "Dotazione finanziaria", "Contributo", "Parole chiave", 
            "Aperto", "Nome file" """,
            output_json=BandoSummaryNew,
            async_execution=False
        )
    
    @staticmethod
    def create_document_analysis_task(agent, business_idea: str, document_context: str) -> Task:
        """Task legacy per compatibilità - reindirizza al nuovo processo iterativo"""
        
        return ReaderTasks.create_iterative_document_analysis_task(
            agent=agent,
            business_idea=business_idea, 
            document_context=document_context,
            filename="Non specificato"  # Fallback se non fornito
        )