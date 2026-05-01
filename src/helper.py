import os
import shutil
import stat
from git import Repo
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Clone any github repo safely
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def repo_ingestion(repo_url):
    repo_path = "repo/"
    
    # If the folder already exists, delete it forcefully
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=remove_readonly)
        
    # Clone the new repository
    Repo.clone_from(repo_url, to_path=repo_path)

# Loading repo as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    return documents

# Creating chunks
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks

# Loading embedding model
def load_embedding():
    # Upgraded to Google's latest embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return embeddings