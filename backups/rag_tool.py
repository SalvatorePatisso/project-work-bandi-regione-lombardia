from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool

class RagSystem:

    def __init__(self, api_key: str, api_end_point: str, api_version: str, embedding_model: str, llm_model: str):
        """
        Initialize the RAG system with a document path and Azure OpenAI credentials.

        Args:
            api_key (str): API key for Azure OpenAI.
            api_version (str): API version for Azure OpenAI.
            embedding_model (str): Name of the Azure OpenAI embedding model to use.
            llm_model (str): Name of the Azure OpenAI LLM model to use.
        """
        self.vector_store = None
        self.loader = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self._prompt = PromptTemplate(
            input_variables=["question", "context"],
            template = """
            You are an AI assistant specialized in funding opportunities provided by the Lombardy Region.
            Your goal is to assist businesses and professionals by providing clear, accurate, and useful information about available grants, eligibility criteria, deadlines, allocated funds, and how to apply.

            Use the documents provided by the system (retrieved_documents) to answer with accuracy and clarity.
            Do not make up information. If the documents do not contain a direct answer, kindly suggest how to obtain official information.

            
            Tone: professional but understandable. Use clear and simple language.
            If the question is vague, kindly ask the user to provide more specific details.
            
            Respond in standard Italian.
            Question: {question}
            Context: {context}
            Answer:
            """

        )
        print(api_key, api_version, embedding_model, llm_model)
        #initilize the embeddings model
        self.embeddings = AzureOpenAIEmbeddings(
            api_key= api_key,
            azure_endpoint = api_end_point,
            api_version=api_version,
            model=embedding_model
        )

        #initialize the llm model
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint = api_end_point,
            model=llm_model,
            temperature=0.7,
            max_tokens=1000
        )

    def _select_loader(self, doc_path: str):
        """
        Select the appropriate document loader based on the file type.
        Args:
            doc_path (str): Path to the document file.
        Returns:
            Document loader instance for the specified file type.
        Errors:
            ValueError: If the file type is not supported.
        """

        if doc_path.endswith('.pdf'):
            return PyPDFLoader(doc_path)
        elif doc_path.endswith('.txt'):
            return TextLoader(doc_path)
        else:
            raise ValueError("Unsupported file type. Please provide a PDF or TXT file.")
    
    def _create_vector_store(self, doc_path: str):
        self.loader = self._select_loader(doc_path)
        documents = self.loader.load_and_split(text_splitter=self.text_splitter)
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        print("Vector store created.")

    def _add_data_file(self,doc_path: str):
        """
        Load and process the document file such as pdf or txt, splitting it into chunks and storing embeddings.
        Args:
            doc_path (str): Path to the document file to be processed.
        """
        # If vector store already exists, just load the document and split it into chunks
        self.loader = self._select_loader(doc_path)
        documents = self.loader.load_and_split(text_splitter=self.text_splitter)
        print(f"Split blog post into {len(documents)} sub-documents.")              #TODO remove this print statement

        self.vector_store.add_documents(documents)

    def add_data_files(self, folder_path: List[str]):
        """
        Load and process multiple document files from a specified folder, splitting them into chunks and storing embeddings.
        Args:
            folder_path (List[str]): List of paths to the document files to be processed.
        """

        #list al documents in the folder
        doc_list = [ os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for doc_path in doc_list:
            if self.vector_store is None:
                self._create_vector_store(doc_path)
            else: 
                if isinstance(doc_path, str):
                    self._add_data_file(doc_path)
                else:
                    raise ValueError("Each document path must be a string.")
            rag.save_vector_store(vector_store_path=folder_path)


    def save_vector_store(self, vector_store_path: str):
        """
        Save the vector store to a specified path.
        Args:
            vector_store_path (str): Path where the vector store will be saved.
        """
        if self.vector_store is not None:
            self.vector_store.save_local(vector_store_path)
            print(f"Vector store saved to {vector_store_path}.")
        else:
            print("No vector store to save. Please add data first.")

    def generate(self, query_text: str, k: int = 5) -> str:
        """
        Query the vector store for relevant documents and generate an answer using the LLM.
        Args:
            query_text (str): The query text to search for relevant documents.
            k (int) = 5: Number of relevant documents to retrieve from the vector store.
        Returns:
            str: The generated answer based on the retrieved documents.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please add data first.")
        
        #build query
        prompt  = self._prompt.invoke({
            "question": query_text,
            "context": self.vector_store.similarity_search(query_text, k=k)
        }) 

        # Generate an answer using the LLM
        answer = self.llm.invoke(prompt)
        
        return answer
    
    def load_vector_store(self, vector_store_path: str):
        """
        Load the vector store from a specified path.
        Args:
            vector_store_path (str): Path from where the vector store will be loaded.
        """
        if self.vector_store is None:
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings,allow_dangerous_deserialization=True)
            print(f"Vector store loaded from {vector_store_path}.")
        else:
            print("Vector store is already initialized.")
    
    def get_retriever_tool(self, k: int = 5):
        """        
        Get a retriever tool for querying the vector store.
        Args:
            k (int): Number of relevant documents to retrieve from the vector store.
        Returns:
            A retriever tool that can be used to query the vector store.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please add data first.")
        else:
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            tool = Tool(
                name="retrieve_documents",
                func=retriever.invoke,
                description="Retrieve relevant documents from the vector store based on a query.",
                return_direct=True
            )
        return tool
    
if __name__ == "__main__":


    LOAD_VECTOR_STORE = True
    # Example usage
    from dotenv import load_dotenv
    import os
    import pathlib
    load_dotenv()

    api_key = os.getenv("AZURE_API_KEY")
    api_end_point = os.getenv("AZURE_API_BASE")
    api_version = os.getenv("AZURE_API_VERSION")
    embedding_model = os.getenv("AZURE_EMBEDDING_MODEL")
    llm_model = os.getenv("AZURE_LLM_MODEL")

    rag = RagSystem(
        api_key=api_key,
        api_end_point=api_end_point,
        api_version=api_version,
        embedding_model=embedding_model,
        llm_model=llm_model
    )
    base_dir = pathlib.Path(__file__).parent.parent
    db_folder = base_dir / "db"
    if not(LOAD_VECTOR_STORE):
        print("Vector store is not initialized. Adding data files...")
        # Add a document file to the RAG system
        # Get the absolute path to the "db" folder under "src"

        rag.add_data_files(folder_path=db_folder)

    else: 
        print("Vector store is already initialized.")
        rag.load_vector_store(vector_store_path=db_folder)


    answer = rag.generate("Riassumi il bando ricircolo STEP e fammi capire come accedere ai benefici previsti in poche parole")
    print(answer.content)

