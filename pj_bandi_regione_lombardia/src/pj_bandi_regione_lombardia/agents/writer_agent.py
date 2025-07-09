# agents/writer_agent.py
from crewai import Agent
from crewai.llm import LLM
import os
import json
import pathlib
from typing import List, Dict
import pandas as pd
from datetime import datetime

class WriterAgent:
    def __init__(self):
        # Configurazione LLM (più semplice, non serve alta precisione)
        self.llm = LLM(
            model=f"azure/{os.getenv('AZURE_LLM_MODEL')}",
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.3,
            max_tokens=2000
        )
        print("✅ WriterAgent LLM configurato")
        
    def read_json_files(self, json_dir: pathlib.Path) -> List[Dict]:
        """Legge tutti i file JSON dalla directory json_description"""
        json_data_list = []
        
        try:
            # Trova tutti i file JSON nella directory
            json_files = list(json_dir.glob("*.json"))
            print(f"📁 Trovati {len(json_files)} file JSON da processare")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Aggiungi il nome del file ai dati
                        data['_filename'] = json_file.name
                        json_data_list.append(data)
                        print(f"   ✅ Letto: {json_file.name}")
                except Exception as e:
                    print(f"   ❌ Errore lettura {json_file.name}: {e}")
                    
            return json_data_list
            
        except Exception as e:
            print(f"❌ Errore nell'accesso alla directory: {e}")
            return []
    
    def map_json_to_excel_columns(self, json_data: Dict) -> Dict:
        """Mappa i dati JSON alle colonne Excel"""
        
        # Definizione delle colonne Excel e mapping con chiavi JSON
        column_mapping = {
            "Ente erogatore": json_data.get("Ente erogatore", ""),
            "Titolo dell'avviso": json_data.get("Titolo dell'avviso", ""),
            "Descrizione aggiuntiva": json_data.get("Descrizione aggiuntiva", ""),
            "Beneficiari": json_data.get("Beneficiari", ""),
            "Apertura": json_data.get("Apertura", ""),
            "Chiusura": json_data.get("Chiusura", ""),
            "Dotazione finanziaria": json_data.get("Dotazione finanziaria", ""),
            "Contributo": json_data.get("Contributo", ""),
            "Note": "",  # Sempre vuoto come richiesto
            "Link": json_data.get("_filename", ""),  # Nome del file JSON
            "Key Words": json_data.get("Parole chiave", ""),  # Mapping diverso
            "Aperto": json_data.get("Aperto", "")
        }
        
        return column_mapping
    
    def create_excel_file(self, json_dir: pathlib.Path, output_filename: str = None) -> str:
        """Crea il file Excel con tutti i dati dei JSON"""
        
        try:
            # Crea la directory excel_output se non esiste
            excel_dir = json_dir.parent / "excel_output"
            excel_dir.mkdir(exist_ok=True)
            print(f"📁 Directory output Excel: {excel_dir}")
            
            # Nome del file Excel
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"bandi_summary_{timestamp}.xlsx"
            
            output_path = excel_dir / output_filename
            
            # Leggi tutti i file JSON
            json_data_list = self.read_json_files(json_dir)
            
            if not json_data_list:
                print("⚠️ Nessun file JSON trovato da processare")
                return None
            
            # Prepara i dati per il DataFrame
            rows = []
            for json_data in json_data_list:
                row = self.map_json_to_excel_columns(json_data)
                rows.append(row)
            
            # Crea il DataFrame
            df = pd.DataFrame(rows)
            
            # Riordina le colonne nell'ordine specificato
            column_order = [
                "Ente erogatore",
                "Titolo dell'avviso",
                "Descrizione aggiuntiva",
                "Beneficiari",
                "Apertura",
                "Chiusura",
                "Dotazione finanziaria",
                "Contributo",
                "Note",
                "Link",
                "Key Words",
                "Aperto"
            ]
            
            df = df[column_order]
            
            # Salva in Excel con formattazione
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Bandi', index=False)
                
                # Ottieni il worksheet per formattazione
                worksheet = writer.sheets['Bandi']
                
                # Formatta l'header
                for cell in worksheet[1]:
                    cell.font = cell.font.copy(bold=True)
                    cell.fill = cell.fill.copy(fgColor="D3D3D3")
                
                # Auto-dimensiona le colonne
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    # Limita la larghezza massima
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"\n✅ File Excel creato: {output_path}")
            print(f"📊 Totale righe: {len(rows)}")
            print(f"📍 PATH COMPLETO: {output_path.absolute()}")
            
            # Mostra un riepilogo
            print("\n📋 Riepilogo bandi processati:")
            for i, row in enumerate(rows, 1):
                print(f"   {i}. {row['Titolo dell\'avviso'][:50]}... ({row['Link']})")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Errore nella creazione del file Excel: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_excel_output(self, excel_path: str) -> bool:
        """Valida che il file Excel sia stato creato correttamente"""
        try:
            path = pathlib.Path(excel_path)
            
            if not path.exists():
                print("❌ File Excel non trovato")
                return False
            
            # Verifica che il file non sia vuoto
            if path.stat().st_size == 0:
                print("❌ File Excel vuoto")
                return False
            
            # Prova a leggere il file per verificare che sia valido
            df = pd.read_excel(path)
            print(f"✅ File Excel valido con {len(df)} righe e {len(df.columns)} colonne")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore validazione Excel: {e}")
            return False
    
    def create_agent(self) -> Agent:
        """Crea l'agente Writer"""
        return Agent(
            role="Excel Report Writer",
            goal="Creare report Excel professionali e ben formattati dai dati JSON estratti",
            backstory="""
            Sei un esperto nella creazione di report Excel professionali.
            La tua specialità è organizzare dati complessi in formati tabellari
            chiari e facilmente consultabili.
            
            Le tue competenze includono:
            - Lettura e parsing di file JSON
            - Creazione di file Excel ben strutturati
            - Formattazione professionale dei dati
            - Validazione della completezza dei dati
            - Gestione di caratteri speciali e formati diversi
            
            Lavori con precisione per garantire che tutti i dati siano
            correttamente mappati nelle colonne appropriate.
            """,
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=3
        )