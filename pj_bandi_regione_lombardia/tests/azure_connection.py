# test_azure_connection.py
"""
Script per testare la connessione ad Azure AI Foundry
Esegui questo script prima di utilizzare CrewAI per verificare la configurazione
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

def test_azure_connection():
    # Carica variabili d'ambiente
    load_dotenv()
    
    # Stampa configurazione (senza mostrare la chiave completa)
    print("=== CONFIGURAZIONE AZURE AI FOUNDRY ===")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    print(f"API Key: {api_key[:8]}...{api_key[-4:] if api_key else 'NON CONFIGURATA'}")
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment}")
    print(f"API Version: {api_version}")
    print()
    
    # Verifica che tutti i parametri siano configurati
    if not all([api_key, endpoint, deployment, api_version]):
        print("❌ ERRORE: Alcune variabili d'ambiente non sono configurate!")
        return False
    
    # Test 1: Connessione con OpenAI nativo
    print("=== TEST 1: OpenAI nativo ===")
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        response = client.chat.completions.create(
            model=deployment,  # Nome del deployment
            messages=[{"role": "user", "content": "Ciao, funzioni?"}],
            max_tokens=50
        )
        
        print(f"✅ Successo: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Errore OpenAI nativo: {e}")
        return False
    
    # Test 2: Connessione con LangChain
    print("\n=== TEST 2: LangChain ===")
    try:
        llm = AzureChatOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            temperature=0.1
        )
        
        response = llm.invoke("Ciao, funzioni?")
        print(f"✅ Successo: {response.content}")
        
    except Exception as e:
        print(f"❌ Errore LangChain: {e}")
        return False
    
    print("\n🎉 TUTTI I TEST SUPERATI! La configurazione Azure AI Foundry è corretta.")
    return True

def get_azure_info():
    """Mostra come ottenere le informazioni necessarie da Azure AI Foundry"""
    print("\n=== COME OTTENERE LE INFORMAZIONI DA AZURE AI FOUNDRY ===")
    print("1. Vai su https://ai.azure.com/")
    print("2. Seleziona il tuo progetto")
    print("3. Vai su 'Deployments' nel menu laterale")
    print("4. Clicca sul tuo deployment GPT-4o")
    print("5. Nella scheda 'Details':")
    print("   - AZURE_OPENAI_ENDPOINT: 'Target URI' (esempio: https://my-resource.openai.azure.com/)")
    print("   - AZURE_OPENAI_DEPLOYMENT_NAME: 'Deployment name' (esempio: gpt-4o-deployment)")
    print("6. Per l'API Key:")
    print("   - Vai su 'Keys and Endpoint' nella risorsa Azure OpenAI")
    print("   - Copia 'KEY 1' o 'KEY 2'")
    print("7. API Version: usa '2024-02-15-preview' (la più recente)")

if __name__ == "__main__":
    print("🔧 TEST CONNESSIONE AZURE AI FOUNDRY")
    print("=" * 50)
    
    if not test_azure_connection():
        print("\n❌ Test fallito! Controlla la configurazione:")
        get_azure_info()
    else:
        print("\n✅ Configurazione corretta! Puoi usare CrewAI con Azure AI Foundry.")