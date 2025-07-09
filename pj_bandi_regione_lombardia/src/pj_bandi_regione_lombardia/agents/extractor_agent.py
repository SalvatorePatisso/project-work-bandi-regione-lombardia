# agents/extractor_agent.py
from crewai import Agent
from crewai.llm import LLM
import os
import json
from typing import Dict, List, Tuple

class ExtractorAgent:
    def __init__(self):
        # Configurazione LLM per estrazione precisa
        self.llm = LLM(
            model=f"azure/{os.getenv('AZURE_LLM_MODEL')}",
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.1,  # Bassa temperatura per maggiore precisione
            max_tokens=4000
        )
        print("✅ ExtractorAgent LLM configurato")
        
    def reconstruct_full_document(self, rag_system, source_file: str) -> str:
        """Ricostruisce il documento completo dai chunk nel vector store"""
        try:
            # Recupera tutti i chunk del documento
            all_chunks = []
            
            # Fai una ricerca molto ampia per recuperare tutti i chunk
            results = rag_system.vector_store.similarity_search(
                query="",  # Query vuota per non filtrare per contenuto
                k=200,  # Numero alto per prendere tutti i chunk
                filter={"source": source_file} if hasattr(rag_system.vector_store, 'filter') else None
            )
            
            # Filtra manualmente se il filter non è supportato
            for doc in results:
                metadata = getattr(doc, 'metadata', {})
                if metadata.get('source') == source_file:
                    all_chunks.append({
                        'content': doc.page_content,
                        'page': metadata.get('page', 0),
                        'metadata': metadata
                    })
            
            # Ordina per numero di pagina
            all_chunks.sort(key=lambda x: x['page'])
            
            # Ricostruisci il documento
            full_document = ""
            current_page = -1
            
            for chunk in all_chunks:
                if chunk['page'] != current_page:
                    full_document += f"\n\n--- PAGINA {chunk['page']} ---\n\n"
                    current_page = chunk['page']
                full_document += chunk['content'] + "\n"
            
            print(f"✅ Documento ricostruito: {len(all_chunks)} chunk, {len(full_document)} caratteri totali")
            return full_document
            
        except Exception as e:
            print(f"❌ Errore nella ricostruzione documento: {e}")
            return ""
    
    def extract_field_with_rag(self, rag_system, field_name: str, field_config: Dict) -> str:
        """
        Estrae un singolo campo usando il RAG system per ricerca mirata
        
        Args:
            rag_system: Il sistema RAG per le ricerche
            field_name: Nome del campo da estrarre
            field_config: Configurazione con query e validazione per il campo
        
        Returns:
            Il valore estratto o "Non specificato"
        """
        try:
            # Usa il RAG per cercare informazioni specifiche
            query = field_config.get('query', field_name)
            rag_response = rag_system.generate(query, k=3)
            
            # Per le date, usa un prompt specializzato
            if field_name in ["Apertura", "Chiusura"]:
                extraction_prompt = f"""
                Dai seguenti contesti, estrai la data di {field_name.lower()} del bando.
                
                CONTESTI TROVATI:
                {rag_response.content}
                
                ISTRUZIONI SPECIFICHE:
                {field_config.get('instruction', f'Estrai il valore di {field_name}')}
                
                IMPORTANTE per l'estrazione delle date:
                1. Se trovi una data in formato testuale (es. "28 marzo 2025", "15 aprile"), convertila in DD/MM/YYYY
                2. Se manca l'anno, deducilo dal contesto (cerca riferimenti come "campagna 2025", "bando 2025-2026")
                3. Mesi in italiano: gennaio=01, febbraio=02, marzo=03, aprile=04, maggio=05, giugno=06, 
                   luglio=07, agosto=08, settembre=09, ottobre=10, novembre=11, dicembre=12
                4. Esempi di conversione:
                   - "28 marzo 2025" → "28/03/2025"
                   - "dal 15 aprile" (in un bando 2025) → "15/04/2025"
                   - "entro il 30 giugno 2025" → "30/06/2025"
                
                FORMATO RICHIESTO: DD/MM/YYYY
                
                Rispondi SOLO con la data nel formato DD/MM/YYYY. Se non trovi la data, rispondi "Non specificato".
                """
            else:
                # Prompt standard per altri campi
                extraction_prompt = f"""
                Dai seguenti contesti, estrai SOLO il valore per "{field_name}".
                
                CONTESTI TROVATI:
                {rag_response.content}
                
                ISTRUZIONI SPECIFICHE:
                {field_config.get('instruction', f'Estrai il valore di {field_name}')}
                
                FORMATO ATTESO:
                {field_config.get('format', 'Testo semplice')}
                
                ESEMPI DI VALORI VALIDI:
                {field_config.get('examples', 'N/A')}
                
                Rispondi SOLO con il valore estratto. Se non trovi l'informazione, rispondi "Non specificato".
                """
            
            response = self.llm.call(extraction_prompt)
            
            # Pulisci la risposta
            value = response.strip()
            
            # Valida il formato se specificato
            if 'validator' in field_config:
                if not field_config['validator'](value):
                    print(f"⚠️ Valore non valido per {field_name}: {value}")
                    return "Non specificato"
            
            return value
            
        except Exception as e:
            print(f"❌ Errore nell'estrazione RAG di {field_name}: {e}")
            return "Non specificato"
    
    def extract_structured_info_hybrid(self, rag_system, full_document: str, filename: str, source_file: str) -> Dict:
        """
        Estrae informazioni usando approccio ibrido: RAG per campi specifici + documento completo per contesto
        
        Args:
            rag_system: Sistema RAG per ricerche mirate
            full_document: Documento completo per contesto generale
            filename: Nome del file
            source_file: Path completo del file sorgente
        """
        
        # Configurazione per ogni campo con query ottimizzate e validatori
        field_configs = {
            "Ente erogatore": {
                "query": "ente erogatore regione lombardia direzione generale amministrazione decreto",
                "instruction": "Identifica l'ente che emette il bando. Cerca 'Regione Lombardia', 'DG', 'Direzione Generale', etc.",
                "examples": "Regione Lombardia, Regione Lombardia - DG Sviluppo Economico, Camera di Commercio Milano",
                "format": "Nome completo dell'ente"
            },
            "Titolo dell'avviso": {
                "query": "titolo avviso bando decreto oggetto denominazione",
                "instruction": "Trova il titolo ufficiale completo del bando. Spesso dopo 'OGGETTO:' o 'AVVISO' o in intestazione",
                "examples": "BANDO SMART WORKING 2024, Avviso pubblico per contributi digitalizzazione PMI",
                "format": "Titolo completo senza abbreviazioni"
            },
            "Apertura": {
                "query": "apertura sportello presentazione domande inizio partire dal giorno marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre gennaio febbraio",
                "instruction": "Cerca la data da cui è possibile presentare domanda. Cerca frasi come: 'a partire dal', 'apertura sportello', 'dalle ore', 'dal giorno', 'presentazione domande dal'. Controlla anche date scritte in forma testuale come '28 marzo 2025', '15 aprile', etc. Se trovi solo mese e giorno, aggiungi l'anno basandoti sul contesto del bando.",
                "examples": "15/01/2024, 01/02/2024, 28/03/2025, dal 15 marzo 2025",
                "format": "DD/MM/YYYY",
                "validator": lambda x: x == "Non specificato" or (len(x.split('/')) == 3 and x[2] == '/')
            },
            "Chiusura": {
                "query": "scadenza termine chiusura presentazione domande entro ultimo giorno ore marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre gennaio febbraio",
                "instruction": "Trova l'ultimo giorno utile per presentare domanda. Cerca: 'entro il', 'termine', 'scadenza', 'fino al', 'chiusura sportello', 'ultimo giorno', 'ore 12:00 del'. Attenzione a date in forma testuale come '30 aprile 2025'. Se trovi solo giorno e mese, deduci l'anno dal contesto.",
                "examples": "31/12/2024, 30/06/2024, 15/09/2024, entro il 30 aprile 2025",
                "format": "DD/MM/YYYY",
                "validator": lambda x: x == "Non specificato" or (len(x.split('/')) == 3 and x[2] == '/')
            },
            "Dotazione finanziaria": {
                "query": "dotazione finanziaria budget stanziamento risorse disponibili totale euro",
                "instruction": "Identifica l'importo totale stanziato per il bando. Includi sempre il simbolo €",
                "examples": "€ 10.000.000, € 5.000.000,00, 2 milioni di euro",
                "format": "Importo con simbolo €"
            },
            "Contributo": {
                "query": "contributo massimo finanziamento importo agevolazione intensità aiuto percentuale",
                "instruction": "Trova tipo e importo massimo del contributo per beneficiario. Specifica se fondo perduto o finanziamento",
                "examples": "Fino a € 100.000 a fondo perduto, 50% delle spese ammissibili max € 200.000",
                "format": "Tipo e importo con €"
            },
            "Beneficiari": {
                "query": "soggetti beneficiari destinatari possono partecipare requisiti PMI startup",
                "instruction": "Chi può partecipare al bando? Elenca tutti i soggetti ammissibili",
                "examples": "PMI lombarde, Micro e piccole imprese, Startup innovative con sede in Lombardia",
                "format": "Elenco soggetti ammissibili"
            }
        }
        
        print("🔄 Inizio estrazione ibrida...")
        extracted_data = {}
        
        # Step 1: Estrai ogni campo usando RAG
        for field_name, config in field_configs.items():
            print(f"   📍 Estrazione {field_name}...")
            value = self.extract_field_with_rag(rag_system, field_name, config)
            extracted_data[field_name] = value
            
            if value != "Non specificato":
                print(f"   ✅ {field_name}: {value[:50]}...")
            else:
                print(f"   ⚠️ {field_name}: Non trovato")
        
        # Step 2: Usa il documento completo per campi derivati
        print("   📍 Elaborazione campi aggiuntivi...")
        
        # Descrizione aggiuntiva - usa il documento completo per sintesi
        if len(full_document) > 500:
            desc_prompt = f"""
            Basandoti su questo estratto del bando e sui dati già estratti, crea una descrizione sintetica (max 150 parole) di cosa finanzia il bando:
            
            ESTRATTO DOCUMENTO:
            {full_document[:2000]}
            
            DATI GIÀ ESTRATTI:
            Ente: {extracted_data.get('Ente erogatore', 'N/A')}
            Titolo: {extracted_data.get('Titolo dell\'avviso', 'N/A')}
            Beneficiari: {extracted_data.get('Beneficiari', 'N/A')}
            
            Descrivi: obiettivi principali, cosa viene finanziato, finalità del bando.
            """
            
            try:
                extracted_data["Descrizione aggiuntiva"] = self.llm.call(desc_prompt).strip()
            except:
                extracted_data["Descrizione aggiuntiva"] = "Bando per il finanziamento di progetti innovativi"
        
        # Parole chiave - estrai dal documento completo
        keywords_prompt = f"""
        Estrai 5-7 parole chiave che caratterizzano questo bando.
        
        TITOLO: {extracted_data.get('Titolo dell\'avviso', 'N/A')}
        BENEFICIARI: {extracted_data.get('Beneficiari', 'N/A')}
        
        ESTRATTO:
        {full_document[:1000]}
        
        Rispondi SOLO con le parole chiave separate da virgola.
        """
        
        try:
            extracted_data["Parole chiave"] = self.llm.call(keywords_prompt).strip()
        except:
            extracted_data["Parole chiave"] = "innovazione, digitalizzazione, PMI, Lombardia"
        
        # Determina se il bando è aperto
        apertura = extracted_data.get("Apertura", "Non specificato")
        chiusura = extracted_data.get("Chiusura", "Non specificato")
        
        if chiusura != "Non specificato":
            try:
                from datetime import datetime
                oggi = datetime.now()
                
                # Converti la data di chiusura
                data_chiusura = datetime.strptime(chiusura, "%d/%m/%Y")
                
                # Confronta con oggi per determinare se è aperto
                if oggi <= data_chiusura:
                    extracted_data["Aperto"] = "si"
                    print(f"   ✅ Bando APERTO (scade il {chiusura})")
                else:
                    extracted_data["Aperto"] = "no"
                    print(f"   ❌ Bando CHIUSO (scaduto il {chiusura})")
                    
            except Exception as e:
                print(f"   ⚠️ Errore nel calcolo stato bando: {e}")
                extracted_data["Aperto"] = "Non specificato"
        else:
            # Se non c'è data di chiusura, non possiamo determinare se è aperto
            extracted_data["Aperto"] = "Non specificato"
            print("   ⚠️ Stato bando non determinabile (manca data chiusura)")
        
        # Aggiungi il nome del file
        extracted_data["Nome file"] = filename
        
        # Step 3: Validazione finale con documento completo - focus su date mancanti
        print("   📍 Validazione finale...")
        
        # Se le date sono ancora "Non specificato", prova una ricerca più ampia
        if extracted_data.get("Apertura") == "Non specificato" or extracted_data.get("Chiusura") == "Non specificato":
            date_search_prompt = f"""
            Nel seguente documento, trova le date di apertura e chiusura per la presentazione delle domande.
            
            DOCUMENTO:
            {full_document[:5000]}
            
            Cerca con attenzione:
            - Date in formato testuale (es. "28 marzo", "15 aprile 2025")
            - Riferimenti a "apertura sportello", "presentazione domande dal"
            - Riferimenti a "scadenza", "termine ultimo", "chiusura sportello"
            - Orari specifici (es. "ore 12:00 del 30 aprile")
            
            Se trovi date in formato testuale, convertile in DD/MM/YYYY.
            Se manca l'anno, deducilo dal contesto (campagna 2025, bando 2025-2026, etc.)
            
            Rispondi in questo formato:
            Apertura: [data in formato DD/MM/YYYY o "Non trovata"]
            Chiusura: [data in formato DD/MM/YYYY o "Non trovata"]
            """
            
            try:
                date_response = self.llm.call(date_search_prompt)
                
                # Parse della risposta per estrarre le date
                lines = date_response.strip().split('\n')
                for line in lines:
                    if 'Apertura:' in line and extracted_data.get("Apertura") == "Non specificato":
                        date_value = line.split(':', 1)[1].strip()
                        if date_value != "Non trovata" and "/" in date_value:
                            extracted_data["Apertura"] = date_value
                            print(f"   🔧 Trovata data apertura: {date_value}")
                    elif 'Chiusura:' in line and extracted_data.get("Chiusura") == "Non specificato":
                        date_value = line.split(':', 1)[1].strip()
                        if date_value != "Non trovata" and "/" in date_value:
                            extracted_data["Chiusura"] = date_value
                            print(f"   🔧 Trovata data chiusura: {date_value}")
            except:
                pass
        
        validation_prompt = f"""
        Verifica questi dati estratti confrontandoli con il documento.
        Se trovi informazioni mancanti o errate, correggile.
        
        DATI ESTRATTI:
        {json.dumps(extracted_data, ensure_ascii=False, indent=2)}
        
        ESTRATTO DOCUMENTO PER VERIFICA:
        {full_document[:3000]}
        
        Rispondi SOLO con il JSON corretto e completo. Non aggiungere spiegazioni.
        """
        
        try:
            response = self.llm.call(validation_prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                validated_data = json.loads(response[json_start:json_end])
                # Mantieni solo i campi che erano "Non specificato" e ora hanno un valore
                for key, value in validated_data.items():
                    if key in extracted_data and extracted_data[key] == "Non specificato" and value != "Non specificato":
                        extracted_data[key] = value
                        print(f"   🔧 Corretto {key}: {value[:50]}...")
        except:
            print("   ⚠️ Validazione fallita, mantengo dati originali")
        
        print("✅ Estrazione ibrida completata")
        return extracted_data
    
    def extract_structured_info(self, full_document: str, filename: str) -> Dict:
        """
        Metodo legacy mantenuto per compatibilità
        """
        # Fallback al metodo originale se chiamato senza RAG
        print("⚠️ Usando metodo legacy senza RAG - risultati potrebbero essere imprecisi")
        return self.extract_structured_info_hybrid(None, full_document, filename, "")
    
    def create_agent(self) -> Agent:
        """Crea l'agente Extractor"""
        return Agent(
            role="Hybrid Precision Document Data Extractor",
            goal="Estrarre con precisione assoluta tutte le informazioni strutturate usando approccio ibrido RAG + documento completo",
            backstory="""
            Sei un esperto analista specializzato nell'estrazione precisa di dati da documenti di bandi pubblici.
            Utilizzi un approccio ibrido che combina:
            - Ricerche RAG mirate per ogni campo specifico
            - Analisi del documento completo per contesto e validazione
            - Pattern matching e validazione dei formati
            
            Le tue competenze includono:
            - Ricerca semantica ottimizzata per ogni tipo di informazione
            - Validazione incrociata dei dati estratti
            - Riconoscimento di pattern tipici dei bandi della Regione Lombardia
            - Correzione automatica di errori comuni
            
            Non inventi mai informazioni e verifichi sempre i dati nel documento.
            """,
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=3
        )