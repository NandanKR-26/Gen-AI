from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os

class RAGPipeline:
    def __init__(self, pdf_path="Data\CUDA_C_Programming_Guide.pdf", persist_dir="./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir
        self.embedding_model = "all-MiniLM-L6-v2"
        self.llm = Ollama(model="mistral")
        self.vectorstore = self._setup_vectorstore()
        self.qa_chain = self._setup_qa_chain()

    def _setup_vectorstore(self):
        if os.path.exists(self.persist_dir):
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=HuggingFaceEmbeddings(model_name=self.embedding_model)
            )
        
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")

        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            texts,
            HuggingFaceEmbeddings(model_name=self.embedding_model),
            persist_directory=self.persist_dir
        )
        vectorstore.persist()
        return vectorstore

    def _setup_qa_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def get_response(self, query):
        result = self.qa_chain({"query": query})
        return result["result"]