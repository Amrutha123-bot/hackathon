import os
import shutil
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Required Libraries ---
# Ensure these are installed: pip install langchain langchain-openai langchain-community pydantic pymupdf unstructured[pdf,docx,eml]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import SelfQueryRetriever
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document # Ensure this is correctly imported

# --- Document Loaders ---
# These need to be available in the environment where this script runs.
# You might need to install 'unstructured' and its dependencies separately if not using pip install "unstructured[...]"
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders import UnstructuredFileLoader # Keep commented unless needed

# --- Configuration ---
load_dotenv() # Load environment variables from a .env file if present

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("ERROR: OPENAI_API_KEY not found. Please set the environment variable.")
    exit(1) # Exit if API key is missing

# Constants
# Note: We use a specific directory for the index generation script.
# This directory will then be committed to your repository.
INDEX_GENERATION_TEMP_DIR = "temp_docs_for_indexing"
CHROMA_DB_DIR = "chroma_db" # This is the directory that will be created and committed.

# --- Pydantic Models for LLM Output (if you need them for parsing within this script) ---
# If you only need this for the main app, you can remove these from here.
# For simplicity in this script, we'll assume the primary goal is just indexing.
# If you want to validate the *source* documents' metadata during index generation, you'd use these.

class QueryDetails(BaseModel):
    """Details extracted from a user query."""
    age: Optional[int] = Field(None, description="Age of the person.")
    procedure: Optional[str] = Field(None, description="Medical procedure or service.")
    location: Optional[str] = Field(None, description="Location related to the query (e.g., city, hospital).")
    policy_duration: Optional[str] = Field(None, description="Duration of the insurance policy (e.g., '3 months', '1 year').")

class DecisionResponse(BaseModel):
    """Structured response for decision making."""
    decision: str = Field(..., description="The final decision (e.g., 'approved', 'rejected', 'pending').")
    amount: Optional[float] = Field(None, description="Payout amount, if applicable.")
    justification: str = Field(..., description="Explanation for the decision, referencing specific clauses.")
    clauses_used: List[str] = Field(..., description="List of document clauses that supported the decision.")


# --- Initialization for LLM/Embeddings ---
# We need these for the SelfQueryRetriever to work properly during indexing if you use it for metadata extraction
try:
    llm_parser_decision = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm_retriever = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    embeddings = OpenAIEmbeddings()
except Exception as e:
    print(f"ERROR initializing OpenAI models: {e}")
    exit(1)


# --- Utility Functions ---

def get_loader_for_file(file_path: str):
    """Returns the appropriate LangChain document loader based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        # PyMuPDFLoader is generally faster and has fewer dependencies than Unstructured for PDFs
        return PyMuPDFLoader(file_path)
    elif file_extension == ".docx":
        return UnstructuredWordDocumentLoader(file_path)
    elif file_extension == ".eml":
        return UnstructuredEmailLoader(file_path)
    elif file_extension == ".txt":
        return TextLoader(file_path)
    else:
        # Fallback for other types if needed, but be aware of unstructured's dependencies
        # try:
        #     from langchain_community.document_loaders import UnstructuredFileLoader
        #     print(f"WARNING: Attempting to load unknown file type '{file_extension}' with UnstructuredFileLoader.")
        #     return UnstructuredFileLoader(file_path)
        # except ImportError:
        #     print("ERROR: langchain-unstructured not installed. Cannot load unknown file types.")
        #     return None
        print(f"WARNING: Unsupported file type '{file_extension}' for file {os.path.basename(file_path)}. Skipping.")
        return None

def load_documents_from_directory_manually(directory_path: str):
    """Loads documents from a directory by iterating through files and using specific loaders."""
    all_docs = []
    files_to_process = []

    if not os.path.exists(directory_path):
        print(f"ERROR: Directory not found: {directory_path}")
        return []

    print(f"Scanning directory: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            files_to_process.append(file_path)

    total_files = len(files_to_process)
    if total_files == 0:
        print("WARNING: No files found in the directory for indexing.")
        return []

    print(f"Found {total_files} files to process.")

    for i, file_path in enumerate(files_to_process):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{total_files}] Loading: {filename}")
        try:
            loader = get_loader_for_file(file_path)
            if loader:
                docs = loader.load()
                if not docs:
                    print(f"WARNING: Loader for {filename} returned empty. Check file content or dependencies.")
                else:
                    for doc in docs:
                        if doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata["source"] = filename # Store original filename as source

                        # --- IMPORTANT: Metadata Extraction for Filtering ---
                        # If you want SelfQueryRetriever to filter by 'age', 'procedure', etc.,
                        # you MUST extract and populate these fields from doc.page_content HERE.
                        # This often requires another LLM call or advanced regex.
                        # For this indexing script, we'll assume you're NOT heavily relying on metadata filtering initially,
                        # or you have pre-extracted metadata elsewhere.
                        # Example (requires importing `re` and `PyMuPDFLoader` if you use its text_extract function):
                        # import re
                        # if file_extension == ".pdf":
                        #     # This is just an example, might need more robust extraction
                        #     pdf_loader = PyMuPDFLoader(file_path)
                        #     page_content = pdf_loader.load()[0].page_content # Example: Get first page content
                        #     age_match = re.search(r"Age:\s*(\d+)", page_content)
                        #     if age_match:
                        #         doc.metadata["age"] = int(age_match.group(1))
                        #     proc_match = re.search(r"Procedure:\s*(.*?)\n", page_content)
                        #     if proc_match:
                        #         doc.metadata["procedure"] = proc_match.group(1).strip()
                        #     # ... and so on for other fields. This can be complex!

                    all_docs.extend(docs)
            else:
                print(f"WARNING: Skipping unsupported file type: {filename}")

        except Exception as e:
            print(f"ERROR loading {filename}: {e}")
            # Continue processing other files

    if not all_docs and files_to_process:
        print("ERROR: Failed to load any documents. Check file formats and ensure necessary dependencies are installed (e.g., pip install 'unstructured[pdf]' python-docx).")
    elif not all_docs and not files_to_process:
        print("WARNING: No files were found to process.")
    else:
        print(f"Successfully loaded {len(all_docs)} document chunks.")

    return all_docs

def initialize_chroma_and_retriever(docs_to_index):
    """Initializes Chroma DB and the SelfQueryRetriever for indexing."""
    if not docs_to_index:
        print("ERROR: No documents provided to index.")
        return False

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = splitter.split_documents(docs_to_index)

    if not split_docs:
        print("ERROR: Failed to split documents. Content might be too short or malformed.")
        return False
    print(f"Split into {len(split_docs)} chunks.")

    print("Creating embeddings and indexing documents into Chroma DB...")

    # --- Define metadata fields for SelfQueryRetriever ---
    # This list tells the retriever which metadata keys it can use for filtering.
    # IMPORTANT: For filtering to work, these fields MUST have been populated in the `doc.metadata`
    # during the load_documents_from_directory_manually step.
    metadata_field_info = [
        {"name": "source", "description": "The source document the chunk came from", "type": "string"},
        # Add other fields here IF you extracted them in load_documents_from_directory_manually
        # Example:
        # {"name": "age", "description": "The age of the policyholder.", "type": "integer"},
        # {"name": "procedure", "description": "The medical procedure performed or service requested.", "type": "string"},
    ]

    try:
        # Clean up any existing Chroma DB directory before creating a new one
        if os.path.exists(CHROMA_DB_DIR):
            print(f"Removing existing Chroma DB directory: {CHROMA_DB_DIR}")
            shutil.rmtree(CHROMA_DB_DIR)

        print(f"Creating Chroma DB in directory: {CHROMA_DB_DIR}")
        # Chroma saves automatically upon creation with persist_directory
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR # Specify directory for Chroma
        )

        # --- Create Retriever ---
        # Note: We are NOT storing the retriever in a global variable here,
        # as this is a script that just creates the index.
        retriever = SelfQueryRetriever.from_llm(
            llm_retriever, # LLM for query understanding
            vectorstore,
            "Brief summary of a document, including its source and any relevant metadata.",
            metadata_field_info, # Metadata definitions for filtering
            verbose=True,
            search_kwargs={'k': 5} # Number of documents to retrieve
        )
        print("Chroma DB and retriever created successfully.")
        return True

    except Exception as e:
        print(f"ERROR during embedding or Chroma DB creation: {e}")
        return False

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting ChromaDB Index Generation ---")

    # 1. Ensure the temporary directory for documents exists and clean it
    if os.path.exists(INDEX_GENERATION_TEMP_DIR):
        print(f"Cleaning existing temp directory: {INDEX_GENERATION_TEMP_DIR}")
        try:
            shutil.rmtree(INDEX_GENERATION_TEMP_DIR)
        except Exception as e:
            print(f"Error clearing {INDEX_GENERATION_TEMP_DIR}: {e}")
    os.makedirs(INDEX_GENERATION_TEMP_DIR)
    print(f"Created temporary directory: {INDEX_GENERATION_TEMP_DIR}")

    # --- IMPORTANT ---
    # You need to place your documents INSIDE the INDEX_GENERATION_TEMP_DIR folder
    # for this script to find them.
    # For example, if you have 'policy1.pdf' and 'policy2.docx',
    # copy them into a folder named 'temp_docs_for_indexing' in the same directory
    # as this script.

    print(f"\nLooking for documents in: {os.path.abspath(INDEX_GENERATION_TEMP_DIR)}")
    print("Please ensure your documents are placed inside this directory.")
    print("Press Enter to continue after placing your documents...\n")
    input() # Wait for user to confirm documents are placed

    # 2. Load documents from the specified temporary directory
    documents_to_index = load_documents_from_directory_manually(INDEX_GENERATION_TEMP_DIR)

    # 3. Initialize Chroma DB and Retriever (this will create the CHROMA_DB_DIR)
    if documents_to_index:
        if initialize_chroma_and_retriever(documents_to_index):
            print("\n--- ChromaDB Index Generation SUCCESSFUL ---")
            print(f"ChromaDB index created at: {os.path.abspath(CHROMA_DB_DIR)}")
            print("You can now commit the 'chroma_db' directory to your repository.")
        else:
            print("\n--- ChromaDB Index Generation FAILED ---")
            exit(1)
    else:
        print("\n--- ChromaDB Index Generation FAILED: No documents were loaded ---")
        exit(1)

    # 4. Clean up the temporary document directory after successful indexing
    print(f"\nCleaning up temporary document directory: {INDEX_GENERATION_TEMP_DIR}")
    try:
        shutil.rmtree(INDEX_GENERATION_TEMP_DIR)
        print("Temporary directory cleaned.")
    except Exception as e:
        print(f"Error cleaning {INDEX_GENERATION_TEMP_DIR}: {e}")

    print("\n--- Index Generation Script Finished ---")