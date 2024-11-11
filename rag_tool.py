import os
import sys
from dotenv import load_dotenv # add pinecone key and jina ai key in .env file.

from pinecone import Pinecone

from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.jinaai import JinaEmbedding
from crewai_tools import tool

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
jina_ai_api_key = os.getenv("JINA_API_KEY")
print("here", pinecone_api_key, jina_ai_api_key)

def get_rag_engine(idx_name = "jina-ai-razorpay-payment-unique"):
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(idx_name)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        add_sparse_vector=True,
    )

    jina_embeddings = JinaEmbedding(api_key=jina_ai_api_key, 
                                    model="jina-embeddings-v3", 
                                    task="retrieval.query",    
                                    embed_batch_size=2, 
                                    dimensions=1024 )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, 
                                                embed_model=jina_embeddings) 

    retriever_engine = index.as_retriever(similarity_top_k=10)

    return retriever_engine

@tool("RAGAgentTool")
def rag_agent_tool(query: str):
    """
    This function retrieves information from a vector database according to the query. 
    Information could be a policy docs, FAQs, process to follow to resolve an issue, etc.
    """
    return [output.text for output in get_rag_engine().retrieve(query)]

if __name__ == "__main__":
    out = get_rag_engine().retrieve("how to enable international payments.")
    #print(out[0].keys())
    for o in out:
        print(o.text)
        print("metadata", o.metadata)
        print("***************************************************")
        pass