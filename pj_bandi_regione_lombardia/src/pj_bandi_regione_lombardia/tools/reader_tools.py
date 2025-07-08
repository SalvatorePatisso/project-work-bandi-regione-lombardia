# tools/reader_tools.py
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json

class LLMDocumentExtractorTool(BaseTool):
    name: str = "LLM Document Extractor Tool"
    description: str = "Utilizza l'LLM per estrarre specifiche informazioni dal documento di bando attraverso domande mirate."
    
    def _run(self, document_text: str, field_name: str) -> str:
        """
        Estrae una specifica informazione dal documento usando l'LLM
        Args:
            document_text: Il testo del documento da analizzare
            field_name: Il nome del campo da estrarre (es. "titolo", "ente_erogatore", etc.)
        """
        
        # Definisce le domande specifiche per ogni campo
        prompts = {
            "ente_erogatore": """
            Analizza il seguente documento di bando e identifica l'ente che eroga il finanziamento.
            Cerca riferimenti a: Regione Lombardia, Camera di Commercio, Ministero, Unioncamere, etc.
            
            Documento:
            {document}
            
            Rispondi SOLO con il nome dell'ente erogatore, senza spiegazioni aggiuntive.
            Se non trovi informazioni chiare, rispondi "Non specificato".
            """,
            
            "titolo_avviso": """
            Analizza il seguente documento di bando e identifica il titolo ufficiale dell'avviso.
            Cerca il titolo principale del bando, spesso scritto in maiuscolo o evidenziato.
            
            Documento:
            {document}
            
            Rispondi SOLO con il titolo del bando, massimo 200 caratteri.
            Se non trovi un titolo chiaro, rispondi "Non specificato".
            """,
            
            "descrizione_aggiuntiva": """
            Analizza il seguente documento di bando e crea una descrizione sintetica (massimo 300 caratteri) 
            che spieghi di cosa si tratta e quali sono gli obiettivi principali del bando.
            
            Documento:
            {document}
            
            Rispondi SOLO con una descrizione concisa, senza introduzioni.
            """,
            
            "beneficiari": """
            Analizza il seguente documento di bando e identifica chi può partecipare al bando.
            Cerca informazioni su: startup, PMI, piccole e medie imprese, cooperative, enti, etc.
            
            Documento:
            {document}
            
            Rispondi SOLO con i tipi di soggetti che possono partecipare, separati da virgola.
            Se non trovi informazioni, rispondi "Non specificato".
            """,
            
            "apertura": """
            Analizza il seguente documento di bando e identifica la data di apertura del bando.
            Cerca date di inizio, apertura, pubblicazione del bando.
            
            Documento:
            {document}
            
            Rispondi SOLO con la data nel formato DD/MM/YYYY.
            Se non trovi la data di apertura, rispondi "Non specificato".
            """,
            
            "chiusura": """
            Analizza il seguente documento di bando e identifica la data di chiusura/scadenza del bando.
            Cerca date di scadenza, termine, chiusura per la presentazione delle domande.
            
            Documento:
            {document}
            
            Rispondi SOLO con la data nel formato DD/MM/YYYY.
            Se non trovi la data di chiusura, rispondi "Non specificato".
            """,
            
            "dotazione_finanziaria": """
            Analizza il seguente documento di bando e identifica la dotazione finanziaria totale disponibile.
            Cerca il budget complessivo, la dotazione, lo stanziamento totale del bando.
            
            Documento:
            {document}
            
            Rispondi SOLO con l'importo (es. "€ 2.000.000" o "2 milioni di euro").
            Se non trovi l'informazione, rispondi "Non specificato".
            """,
            
            "contributo": """
            Analizza il seguente documento di bando e identifica il tipo e l'importo del contributo erogabile.
            Cerca informazioni su: fondo perduto, finanziamento agevolato, percentuale di copertura, importo massimo.
            
            Documento:
            {document}
            
            Rispondi SOLO con il tipo di contributo e l'importo (es. "Fondo perduto fino a € 150.000").
            Se non trovi informazioni, rispondi "Non specificato".
            """,
            
            "parole_chiave": """
            Analizza il seguente documento di bando e identifica le 5-7 parole chiave principali 
            che descrivono i settori, le tecnologie o gli ambiti di intervento del bando.
            
            Documento:
            {document}
            
            Rispondi SOLO con le parole chiave separate da virgola (es. "innovazione, digitale, sostenibilità, IoT").
            """,
            
            "aperto": """
            Analizza il seguente documento di bando e determina se il bando è attualmente aperto o chiuso.
            Confronta le date attuali con le scadenze indicate nel documento.
            
            Documento:
            {document}
            
            Rispondi SOLO con "si" se il bando è aperto o "no" se è chiuso.
            Se non puoi determinarlo, rispondi "Non specificato".
            """
        }
        
        # Ottiene il prompt specifico per il campo richiesto
        if field_name not in prompts:
            return f"Errore: Campo '{field_name}' non riconosciuto"
        
        prompt = prompts[field_name].format(document=document_text)
        
        # Restituisce il prompt formattato (sarà elaborato dall'LLM dell'agente)
        return prompt

class BusinessAlignmentTool(BaseTool):
    name: str = "Business Alignment Tool"
    description: str = "Valuta l'allineamento tra l'idea di business dell'utente e il bando analizzato."
    
    def _run(self, business_idea: str, document_text: str) -> str:
        """Valuta l'allineamento tra business idea e bando usando l'LLM"""
        
        prompt = f"""
        Analizza l'allineamento tra l'idea di business dell'utente e il bando fornito.
        
        IDEA DI BUSINESS:
        {business_idea}
        
        DOCUMENTO BANDO:
        {document_text}
        
        Valuta l'allineamento considerando:
        - Settori target del bando vs settore dell'idea
        - Beneficiari ammessi vs tipologia di azienda
        - Obiettivi del bando vs obiettivi dell'idea
        - Tecnologie/innovazioni richieste vs proposte
        
        Fornisci:
        1. Un punteggio da 0 a 100
        2. Una spiegazione breve dell'allineamento
        3. Raccomandazioni specifiche
        
        Formato della risposta:
        Punteggio: [0-100]
        Spiegazione: [spiegazione breve]
        Raccomandazioni: [2-3 raccomandazioni]
        """
        
        return prompt

class JsonBuilderTool(BaseTool):
    name: str = "JSON Builder Tool"
    description: str = "Costruisce il JSON finale con tutte le informazioni estratte dal documento."
    
    def _run(self, extracted_data: str) -> str:
        """
        Costruisce il JSON finale con i dati estratti
        Args:
            extracted_data: Stringa contenente tutti i dati estratti dall'LLM
        """
        
        prompt = f"""
        Basandoti sui seguenti dati estratti dal documento di bando, crea un JSON strutturato 
        con ESATTAMENTE queste chiavi (rispetta maiuscole/minuscole e spazi):

        DATI ESTRATTI:
        {extracted_data}
        
        Crea un JSON con questa struttura esatta:
        {{
            "Ente erogatore": "valore",
            "Titolo dell'avviso": "valore", 
            "Descrizione aggiuntiva": "valore",
            "Beneficiari": "valore",
            "Apertura": "valore",
            "Chiusura": "valore", 
            "Dotazione finanziaria": "valore",
            "Contributo": "valore",
            "Parole chiave": "valore",
            "Aperto": "valore",
            "Nome file": "valore"
        }}
        
        IMPORTANTE:
        - Usa ESATTAMENTE le chiavi indicate (con spazi e maiuscole)
        - Se un'informazione non è disponibile, usa "Non specificato"
        - Per "Aperto" usa solo "si" o "no"
        - Il JSON deve essere valido e formattato correttamente
        
        Rispondi SOLO con il JSON, senza testo aggiuntivo.
        """
        
        return prompt