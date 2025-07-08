# test_llm_connection.py
"""
Script per testare la connessione LLM di CrewAI con Azure
"""
import os
from dotenv import load_dotenv
from crewai.llm import LLM

def test_crewai_llm():
    load_dotenv()
    
    print("=== TEST CONNESSIONE CREWAI LLM ===")
    
    # Mostra configurazione
    print(f"AZURE_API_KEY: {os.getenv('AZURE_API_KEY')[:10]}...")
    print(f"AZURE_API_BASE: {os.getenv('AZURE_API_BASE')}")
    print(f"AZURE_LLM_MODEL: {os.getenv('AZURE_LLM_MODEL')}")
    print(f"AZURE_API_VERSION: {os.getenv('AZURE_API_VERSION')}")
    print()
    
    try:
        # Test 1: Configurazione base
        print("Test 1: Configurazione base...")
        llm = LLM(
            model=os.getenv("AZURE_LLM_MODEL"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.3
        )
        
        response = llm.call("Ciao, funzioni?")
        print(f"✅ Test 1 riuscito: {response}")
        
    except Exception as e:
        print(f"❌ Test 1 fallito: {e}")
        
        # Test 2: Con azure/ prefix
        try:
            print("\nTest 2: Con prefisso azure/...")
            llm2 = LLM(
                model=f"azure/{os.getenv('AZURE_LLM_MODEL')}",
                api_key=os.getenv("AZURE_API_KEY"),
                api_base=os.getenv("AZURE_API_BASE"),
                api_version=os.getenv("AZURE_API_VERSION"),
                temperature=0.3
            )
            
            response2 = llm2.call("Ciao, funzioni?")
            print(f"✅ Test 2 riuscito: {response2}")
            
        except Exception as e2:
            print(f"❌ Test 2 fallito: {e2}")
            
            # Test 3: Configurazione alternativa
            try:
                print("\nTest 3: Configurazione variabili d'ambiente...")
                
                # Imposta variabili per CrewAI
                os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_API_KEY")
                os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_API_BASE")
                os.environ["OPENAI_API_TYPE"] = "azure"
                os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
                
                llm3 = LLM(
                    model=os.getenv("AZURE_LLM_MODEL"),
                    temperature=0.3
                )
                
                response3 = llm3.call("Ciao, funzioni?")
                print(f"✅ Test 3 riuscito: {response3}")
                
            except Exception as e3:
                print(f"❌ Test 3 fallito: {e3}")
                print("\n🔧 SUGGERIMENTI:")
                print("1. Verifica che il deployment 'gpt-4o' esista in Azure")
                print("2. Controlla che l'endpoint sia corretto")
                print("3. Verifica che la API key sia valida")

if __name__ == "__main__":
    test_crewai_llm()