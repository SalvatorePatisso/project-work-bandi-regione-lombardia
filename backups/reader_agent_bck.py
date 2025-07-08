# agents/reader_agent.py
from crewai import Agent
from crewai.llm import LLM
from tools.reader_tools import DocumentAnalysisTool, BusinessAlignmentTool
from tools.rag_tool import RagSystem
import os
from dotenv import load_dotenv
from config.config import vector_store_path

load_dotenv()

class ReaderAgent:
    def __init__(self):
        # Configurazione Azure OpenAI con crewai.llm.LLM
        self.llm = LLM(
            model="azure/gpt-4o",  # Formato: azure/nome-modello
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_LLM_ENDPOINT"),
            api_version=os.getenv("AZURE_LLM_API_VERSION"),
            temperature=0.3,
            max_tokens=4000
        )
        
        # Inizializzazione tools
        self.document_analysis_tool = DocumentAnalysisTool()
        self.business_alignment_tool = BusinessAlignmentTool()
        self.rag_system = RagSystem(
            api_key=os.getenv("AZURE_API_KEY"),
            api_end_point=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            embedding_model=os.getenv("AZURE_EMBEDDING_MODEL"),
            llm_model=os.getenv("AZURE_LLM_MODEL")
        )
        self.rag_system.load_vector_store(vector_store_path=vector_store_path)
  
    def create_agent(self) -> Agent:
        """Crea e configura l'agente Reader con CrewAI LLM"""
        
        return Agent(
            role="Document Reader and Business Analyst",
            goal="Analizzare documenti di bandi e fornire informazioni essenziali allineate alle esigenze di business dell'utente",
            backstory="""
            Sei un esperto analista di bandi pubblici e finanziamenti aziendali. 
            La tua specialità è identificare rapidamente le informazioni chiave nei documenti 
            di bandi e correlarle con le esigenze specifiche di business degli utenti.
            
            Hai anni di esperienza nell'analisi di documenti tecnici e nella sintesi 
            di informazioni complesse in formati facilmente comprensibili.
            """,
            verbose=True,
            allow_delegation=False,
            tools=[
                self.document_analysis_tool,
                self.business_alignment_tool,
                self.rag_system.get_retriever_tool(k=5)
            ],
            llm=self.llm,
            max_iter=3,
            memory=True
        )