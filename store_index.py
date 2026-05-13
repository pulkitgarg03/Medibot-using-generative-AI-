from src.helper import load_pdf_files, create_chunks, get_embeddings_model  
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from dotenv import dotenv_values

config = dotenv_values(".env")
PINECONE_API_KEY = config["PINECONE_API_KEY"]


load_dotenv()

documents = load_pdf_files(data='Data/')
text_chunks = create_chunks(documents)
embedding_model = get_embeddings_model()

pc = pinecone(api_key=PINECONE_API_KEY)
index_name = "medibot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents= text_chunks,
    index_name = index_name,
    embedding= embedding_model
)