from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

#Extract Data From the PDF File
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documents = load_pdf_files(data='Data/')
print("Length of PDF pages", len(documents))


#Split the Data into Text Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap = 30)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(documents)
print("Length of Text Chunks", len(text_chunks))

# Embedding from HuggingFace
def get_embeddings_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embeddings_model()