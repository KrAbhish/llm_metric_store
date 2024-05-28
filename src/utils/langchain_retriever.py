from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

class LangchainRetriever():
    def __init__(self, embedding, ) -> None:
        self.embeddings = embedding
        self.vector_store = Chroma(collection_name="full_documents", embedding_function=self.embeddings)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        self.docstore = InMemoryStore()
        self.parent_retriever = ParentDocumentRetriever(vectorstore=self.vector_store, docstore=self.docstore, child_splitter=self.child_splitter)

    def index_documents(self, documents: List[Document]):
        self.parent_retriever.add_documents(documents, ids=None)

    def retrieve_sub_document(self, query: str):
        sub_docs = self.vector_store.similarity_search(query)
        return sub_docs

    def retrieve_document(self, query: str ):
        retrieved_docs = self.parent_retriever.invoke(query)
        return retrieved_docs
    


    

