import os
import json
from typing import Sequence
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.tools.types import ToolMetadata
from llama_index.core import QueryBundle

import logging

from llama_index.legacy.vector_stores import QdrantVectorStore
from llama_index.legacy import LLMPredictor, GPTVectorStoreIndex, Document, ServiceContext
# from llama_index.legacy import LLMPredictor, GPTVectorStoreIndex, ServiceContext
from llama_index.legacy.storage import StorageContext
import qdrant_client
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from qdrant_client.http.exceptions import UnexpectedResponse


class LlamaIndexRetriever():

    def __init__(
        self,
        # llm:AzureChatOpenAI,
        # embedding_llm: AzureOpenAIEmbeddings,
        collection_name:str,
    ):
        self.collection_name = collection_name
        # self.llm = llm
        # self.embedding_llm = embedding_llm
        self.client = qdrant_client.QdrantClient(":memory:", timeout=360)
        self.aclient = qdrant_client.AsyncQdrantClient(":memory:", timeout=360)


    def initiliaze_context(self,async_client=False):
        if async_client:
            vector_store = QdrantVectorStore(aclient= self.aclient, collection_name=self.collection_name)
        else:
            vector_store = QdrantVectorStore(client= self.client, collection_name=self.collection_name)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embeddings =  AzureOpenAIEmbedding(
                deployment=os.getenv("EMB_DEPLOYMENT"),
                openai_api_version=os.getenv("EMB_OPENAI_API_VERSION"),
                model=os.getenv("EMB_MODEL"),
                api_key=os.getenv("EMB_OPENAI_API_KEY"),
                openai_api_base=os.getenv("EMB_OPENAI_ENDPOINT"),
                openai_api_type=os.getenv("EMB_API_TYPE"),
            )

        llm_gpt = AzureOpenAI(deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'), openai_api_version=os.getenv("OPENAI_API_VERSION"),
                        openai_api_base=os.getenv("OPENAI_API_BASE"), 
                        openai_api_type= os.getenv("OPENAI_API_TYPE"),
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        max_tokens=3500,
                        temperature=0.0)
        llm_predictor = LLMPredictor(llm=llm_gpt)

        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            embed_model=embeddings,
        )

        return storage_context, service_context
    
    def index_document(self, cube_docs, **kwargs: any):
        storage_context, service_context= self.initiliaze_context()
        llama_docs = [Document(text=i.page_content, metadata=i.metadata) for i in cube_docs]

        document_index = GPTVectorStoreIndex.from_documents(llama_docs, storage_context=storage_context,
                                                            service_context=service_context,
                                                            )
        return document_index
    
    def document_search_setup(self, documents, async_client=False):
        storage_context, service_context = self.initiliaze_context()
        document_index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                                            service_context=service_context, async_client=async_client)
        query_engine = document_index.as_query_engine()
        return query_engine
    
    def query_util(self, question, documents, retry=0):
    
        query_engine = self.document_search_setup(documents, async_client=False)
        try:
            response = query_engine.retrieve(question)
            
            return response
        except UnexpectedResponse as e:
            logging.error(f"Qdrant async client error {e}")
            logging.error(f"Got empty string from Qdrant. Retrying qdrant request - {retry}")
            retry += 1
            # Execute retry 3 times before failing the request
            if retry < 3:
                return self.query_util(query_engine, question, retry)