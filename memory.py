from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

## loading data 
data_path="Data/"
def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents
docs=load_pdf_files(data_path)
print(len(docs))

## creating chunks 
def create_chunks(extracted_data):
    splitter=RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=50,
                separators=' '
    )
    chunks=splitter.split_documents(extracted_data)
    return chunks
chunks=create_chunks(docs)

## generating embeddings
def get_embedd_model():
    model=HuggingFaceEndpointEmbeddings(
        repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    return model
emb_model=get_embedd_model()

db_faiss_path="vectorstore/db_faiss"
db=FAISS.from_documents(chunks,emb_model)
db.save_local(db_faiss_path)




