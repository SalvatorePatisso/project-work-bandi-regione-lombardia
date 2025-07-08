from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

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
        self.text_splitter = None
        self._prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are a helpful assistant. Use the context provided to answer the question.
            Question: {question}
            Context: {context}
            Answer:"""
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
        
    def add_data_file(self,doc_path: str):
        """
        Load and process the document file such as pdf or txt, splitting it into chunks and storing embeddings.
        Args:
            doc_path (str): Path to the document file to be processed.
        """
        if self.vector_store is None:
            #Select loader based on file type
            self.loader = self._select_loader(doc_path)
        
            # Initialize the text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            # Load the document and split it into chunks
            documents = self.loader.load_and_split(text_splitter=self.text_splitter)
            print(f"Split blog post into {len(documents)} sub-documents.")


            # Store the embeddings in the vector store
            self.vector_store = FAISS.from_documents(
                documentsA,
                self.embeddings
            )
            
        else:
            # If vector store already exists, just load the document and split it into chunks
            self.loader = self._select_loader(doc_path)
            documents = self.loader.load_and_split(text_splitter=self.text_splitter)
            print(f"Split blog post into {len(documents)} sub-documents.")

            self.vector_store.add_documents(documents)
    
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

        print(f"Generated answer: {answer}")
        
        return answer

    