# tasks/extractor_tasks.py
from crewai import Task
from pydantic import BaseModel, Field

class BandoSummaryNew(BaseModel):
    """Modello per le informazioni essenziali del bando"""
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

class ExtractorTasks:
    """Classe per la gestione dei task dell'agente Extractor"""
    
    @staticmethod
    def create_full_document_extraction_task(agent, full_document: str, filename: str) -> Task:
        """
        Task per l'estrazione completa di informazioni dal documento intero
        
        Args:
            agent: L'agente Extractor
            full_document: Il documento completo ricostruito
            filename: Nome del file del documento
        
        Returns:
            Task configurato per l'estrazione
        """
        
        return Task(
            description=f"""
            Analizza il seguente documento COMPLETO di bando ed estrai TUTTE le informazioni richieste con precisione assoluta.
            
            DOCUMENTO COMPLETO DA ANALIZZARE:
            {full_document}
            
            NOME FILE: {filename}
            
            PROCESSO DI ESTRAZIONE SISTEMATICA:
            
            1. ENTE EROGATORE:
               - Cerca: "Regione Lombardia", "Camera di Commercio", "Ministero", etc.
               - Posizioni tipiche: intestazione, prima pagina, sezione "Soggetto proponente"
            
            2. TITOLO DELL'AVVISO:
               - Cerca: "Avviso", "Bando", "Decreto", "Oggetto"
               - Di solito in grassetto o dopo "OGGETTO:"
            
            3. DESCRIZIONE AGGIUNTIVA:
               - Sintetizza in max 200 parole cosa finanzia il bando
               - Focus su: obiettivi, attività ammissibili, finalità
            
            4. BENEFICIARI:
               - Cerca: "Soggetti beneficiari", "Destinatari", "Possono partecipare"
               - Specifica: PMI, startup, grandi imprese, requisiti dimensionali
            
            5. DATE APERTURA E CHIUSURA:
               - Apertura: "a partire dal", "dalle ore", "apertura sportello"
               - Chiusura: "entro il", "fino al", "termine ultimo", "scadenza"
               - Formato richiesto: DD/MM/YYYY
            
            6. DOTAZIONE FINANZIARIA:
               - Cerca: "dotazione", "budget", "stanziamento", "risorse disponibili"
               - Include sempre il simbolo €
            
            7. CONTRIBUTO:
               - Cerca: "contributo massimo", "intensità di aiuto", "finanziamento"
               - Specifica: fondo perduto, finanziamento agevolato, percentuale
            
            8. PAROLE CHIAVE:
               - Estrai 5-7 termini chiave che caratterizzano il bando
               - Focus su: settore, tecnologie, obiettivi
            
            9. STATO (APERTO/CHIUSO):
               - Confronta la data odierna con le date di apertura/chiusura
               - Rispondi solo "si" o "no"
            
            REGOLE CRITICHE:
            - MAI inventare informazioni non presenti
            - Se un dato non c'è, usa "Non specificato"
            - Mantieni la formattazione originale per importi e date
            - Verifica ogni informazione nel documento
            
            OUTPUT RICHIESTO:
            Un JSON valido e completo con TUTTE le chiavi richieste.
            """,
            agent=agent,
            expected_output="""JSON strutturato con precisione assoluta contenente:
            "Ente erogatore", "Titolo dell'avviso", "Descrizione aggiuntiva", "Beneficiari", 
            "Apertura", "Chiusura", "Dotazione finanziaria", "Contributo", "Parole chiave", 
            "Aperto", "Nome file". Ogni campo deve essere estratto dal documento senza inventare.""",
            output_json=BandoSummaryNew,
            async_execution=False
        )
    
    @staticmethod
    def create_validation_task(agent, extracted_data: dict, full_document: str) -> Task:
        """
        Task per validare i dati estratti confrontandoli con il documento
        
        Args:
            agent: L'agente Extractor
            extracted_data: Dati già estratti da validare
            full_document: Documento completo per verifica
        
        Returns:
            Task configurato per la validazione
        """
        
        return Task(
            description=f"""
            Valida i seguenti dati estratti confrontandoli con il documento originale.
            
            DATI DA VALIDARE:
            {extracted_data}
            
            DOCUMENTO ORIGINALE:
            {full_document[:5000]}...
            
            PROCESSO DI VALIDAZIONE:
            
            Per ogni campo estratto:
            1. Verifica che l'informazione sia effettivamente presente nel documento
            2. Controlla che non ci siano errori di interpretazione
            3. Assicurati che le date siano nel formato corretto
            4. Verifica che gli importi includano la valuta
            5. Controlla che non ci siano campi con "Non specificato" che invece sono presenti
            
            Se trovi errori o imprecisioni:
            - Segnala quali campi sono errati
            - Fornisci la correzione basata sul documento
            - Spiega brevemente il motivo della correzione
            
            OUTPUT:
            - "VALIDATO" se tutti i dati sono corretti
            - Lista delle correzioni necessarie se ci sono errori
            """,
            agent=agent,
            expected_output="Report di validazione con eventuali correzioni necessarie",
            async_execution=False
        )
    
    @staticmethod
    def create_section_extraction_task(agent, full_document: str, section_name: str) -> Task:
        """
        Task per estrarre informazioni da una sezione specifica del documento
        
        Args:
            agent: L'agente Extractor
            full_document: Il documento completo
            section_name: Nome della sezione da analizzare
        
        Returns:
            Task per estrazione di sezione specifica
        """
        
        return Task(
            description=f"""
            Estrai tutte le informazioni rilevanti dalla sezione "{section_name}" del documento.
            
            DOCUMENTO:
            {full_document}
            
            SEZIONE DA ANALIZZARE: {section_name}
            
            1. Localizza la sezione nel documento
            2. Estrai TUTTO il contenuto di quella sezione
            3. Identifica informazioni chiave come:
               - Requisiti
               - Criteri
               - Importi
               - Scadenze
               - Procedure
            
            4. Struttura le informazioni in modo chiaro e organizzato
            
            Se la sezione non esiste, indicalo chiaramente.
            """,
            agent=agent,
            expected_output=f"Contenuto completo e strutturato della sezione {section_name}",
            async_execution=False
        )