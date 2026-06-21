import os
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# Load environment variables (ensure you have a .env file with FIREWORKS_API_KEY)
load_dotenv()

# API Configurations
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL_NAME = "accounts/fireworks/models/deepseek-v4-pro"

# Request Configuration
DEFAULT_PARAMS = {
    "max_tokens": 4096,
    "top_p": 1,
    "top_k": 40,
    "temperature": 0.6,
}

print("Initializing Corporate Knowledge Copilot Infrastructure...")

# 1. Setup local company document pool directory
DOCS_DIR = "./my docs"
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    with open(f"{DOCS_DIR}/company_policy.txt", "w") as f:
        f.write("The 2026 corporate travel policy allows a maximum hotel budget of $200 per night.\n"
                "All international flights must be pre-approved by the department head 14 days in advance.")

# Setup for temporary uploads (used by extraction_node)
TEMP_DIR = "./temp_uploads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

print(" -> Loading local company documentation files...")
loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
document_chunks = text_splitter.split_documents(raw_documents)

print(" -> Generating embeddings and indexing chunks into Chroma...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents=document_chunks, embedding=embedding_model)

# NOTE: OllamaLLM has been removed.
# You will now use your custom 'call_deepseek' function (in llm_service.py)
# instead of an 'llm' object here.

print(" -> Booting up Cross-Encoder Re-ranker model...")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print(" -> Activating external web search fallback engine...")

def web_search_tool(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                results = list(ddgs.answers(query))
            if not results:
                return "Alternative Web Search Context: Active real-time updates are processing."

            combined_text = []
            for r in results:
                body_text = r.get('body') or r.get('text') or ""
                title_text = r.get('title') or "Web Result"
                if body_text:
                    combined_text.append(f"Source: {title_text} - Data: {body_text}")

            return "\n\n".join(combined_text) if combined_text else "Search yielded no clear description."
    except Exception as e:
        return f"Fallback Context Payload: Real-time records for '{query}' are unavailable."

print("--- ALL INFRASTRUCTURE PIECES ONLINE ---")