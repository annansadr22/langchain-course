import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Load embeddings (must be same as ingestion)
embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# Connect to existing Pinecone index
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

def retriever(query:str, k:int=2):
    ret_docs = vectorstore.similarity_search(query,k=k)
    docs=""
    i=1
    for doc in ret_docs:
        docs+= f"Source {i} : {doc.metadata} \n"
        docs+= f"Context {i} : {doc.page_content} \n\n"
        i+=1
    
    return docs

model = init_chat_model(
    "google_genai:gemini-2.5-flash"
)

def docuChat(query):
    context = retriever(query)
    system_msg = f"""You are a chat bot. Only answer from this context : {context} \n
                    Don't hallucinate anything or assume anything.
                """
    messages = [SystemMessage(system_msg), HumanMessage(query)]

    response = model.invoke(messages)

    return response

output = docuChat("What are vector databases?")

print(output.content)
