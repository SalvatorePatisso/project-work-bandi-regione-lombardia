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

class BandoSummary(BaseModel):
    """Modello per le informazioni essenziali del bando"""
    title: str = Field(..., description="Titolo del bando")
    description: str = Field(..., description="Descrizione sintetica")
    funding_amount: str = Field(..., description="Importo finanziamento disponibile")
    eligibility_criteria: List[str] = Field(..., description="Criteri di eleggibilità principali")
    application_deadline: str = Field(..., description="Scadenza presentazione domanda")
    key_requirements: List[str] = Field(..., description="Requisiti chiave")
    alignment_score: float = Field(..., description="Punteggio allineamento con idea business (0-100)")
    alignment_explanation: str = Field(..., description="Spiegazione dell'allineamento")
    next_steps: List[str] = Field(..., description="Prossimi passi consigliati")

class ReaderTasks:
    
    @staticmethod
    def create_document_analysis_task(agent, business_idea: str, document_context: str) -> Task:
        """Task per analizzare il documento e estrarre informazioni chiave"""
        
        return Task(
            description=f"""
            Analizza il seguente documento di bando fornito tramite RAG e l'idea di business dell'utente:
            
            IDEA DI BUSINESS:
            {business_idea}
            
            DOCUMENTO BANDO (fornito da RAG):
            {document_context}
            
            Il tuo compito è:
            1. Analizzare il documento del bando identificando le informazioni chiave
            2. Valutare l'allineamento tra l'idea di business e i requisiti del bando
            3. Estrarre le informazioni essenziali richieste
            4. Calcolare un punteggio di allineamento (0-100)
            5. Fornire raccomandazioni sui prossimi passi
            6. Generare un riassunto strutturato finale basato sui risultati dei tool
            
            Concentrati su:
            - Criteri di eleggibilità
            - Importi di finanziamento
            - Scadenze
            - Requisiti specifici
            - Settori target
            - Dimensioni aziendali ammissibili
            
            IMPORTANTE: 
            1. Prima usa DocumentAnalysisTool per analizzare il documento
            2. Poi usa BusinessAlignmentTool per valutare l'allineamento
            3. Infine, sulla base dei risultati dei tool, genera tu stesso un riassunto strutturato in formato JSON che includa:
               - title: titolo del bando
               - description: descrizione sintetica
               - funding_amount: importo finanziamento
               - eligibility_criteria: criteri di eleggibilità (max 5)
               - application_deadline: scadenza
               - key_requirements: requisiti chiave (max 5)
               - alignment_score: punteggio allineamento
               - alignment_explanation: spiegazione allineamento
               - next_steps: prossimi passi consigliati
            """,
            agent=agent,
            expected_output="Riassunto strutturato in formato JSON delle informazioni essenziali del bando",
            output_json=BandoSummary,
            async_execution=False
        )
    
    @staticmethod
    def create_summary_generation_task(agent, analysis_result: str) -> Task:
        """Task per generare il riassunto delle informazioni essenziali"""
        
        return Task(
            description=f"""
            Basandoti sull'analisi precedente, genera un riassunto strutturato delle informazioni essenziali:
            
            RISULTATO ANALISI:
            {analysis_result}
            
            Genera un riassunto che includa:
            1. Titolo e descrizione sintetica del bando
            2. Importo del finanziamento disponibile
            3. Criteri di eleggibilità principali (massimo 5 punti)
            4. Scadenza per la presentazione
            5. Requisiti chiave (massimo 5 punti)
            6. Punteggio di allineamento con l'idea di business (0-100)
            7. Spiegazione dell'allineamento
            8. Prossimi passi consigliati (massimo 3 punti)
            
            Il riassunto deve essere chiaro, conciso e orientato all'azione.
            """,
            agent=agent,
            expected_output="Riassunto strutturato in formato JSON delle informazioni essenziali",
            output_json=BandoSummary,
            async_execution=False
        )