import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()




print("Hello from langchain-course!")
file_path = r'C:\Users\NXTWAVE\Documents\Udemy\Langchain(Eden)\langchain-course\file1.txt'
loader = TextLoader(file_path, encoding='UTF-8')
document = loader.load()

print("Splitting")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)
print(f"Created {len(texts)} chunks")


print("Embedding")

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2",
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

print('Ingesting...')
vectorstore = PineconeVectorStore.from_documents(
    documents=texts,
    embedding=embeddings,
    index_name=os.getenv('INDEX_NAME')
)

print('Finished')

