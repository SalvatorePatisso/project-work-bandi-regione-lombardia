# main.py
from crewai import Crew, Process
from agents.reader_agent import ReaderAgent
from tasks.reader_tasks import ReaderTasks
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from tools.rag_tool import RagSystem

def main():
    # Carica variabili d'ambiente
    load_dotenv()
    
    # Simula input utente e contesto RAG
    business_idea = """
    Voglio sviluppare una piattaforma digitale per la gestione sostenibile 
    delle risorse idriche in agricoltura, utilizzando sensori IoT e AI per 
    ottimizzare l'irrigazione e ridurre gli sprechi. La mia azienda è una 
    startup innovativa nel settore agritech.
    """
    
    # Simula documento recuperato tramite RAG
    document_context = """
    BANDO REGIONALE PER L'INNOVAZIONE DIGITALE IN AGRICOLTURA
    
    Il presente bando è finalizzato a supportare progetti di innovazione 
    digitale nel settore agricolo, con particolare focus su:
    - Tecnologie IoT per il monitoraggio delle colture
    - Sistemi di intelligenza artificiale per l'ottimizzazione dei processi
    - Soluzioni per la sostenibilità ambientale
    
    BENEFICIARI:
    - Startup innovative
    - Piccole e medie imprese del settore agritech
    - Imprese agricole che adottano tecnologie digitali
    
    FINANZIAMENTO:
    - Contributo a fondo perduto fino a €150.000
    - Copertura fino al 70% dei costi ammissibili
    
    REQUISITI:
    - Sede operativa nella regione
    - Progetto di durata massima 24 mesi
    - Componente innovativa dimostrabile
    - Partnership con enti di ricerca (preferibile)
    
    SCADENZA: 31 marzo 2025
    """
    # Inizializza agente e task
    reader_agent_instance = ReaderAgent()
    reader_agent = reader_agent_instance.create_agent()
    
    # Crea task
    analysis_task = ReaderTasks.create_document_analysis_task(
        agent=reader_agent,
        business_idea=business_idea,
        document_context=document_context
    )
    
    # Crea crew
    crew = Crew(
        agents=[reader_agent],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Esegui crew
    result = crew.kickoff()
    
    print("=== RISULTATO READER AGENT ===")
    print(result)
    
    return result

if __name__ == "__main__":
    main()