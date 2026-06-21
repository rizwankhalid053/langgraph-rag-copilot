from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Import your new API service
from llm_service import call_deepseek
# Import config
from config import vector_store, rerank_model, web_search_tool


# 1. DEFINE STATE
class GraphState(TypedDict):
    query: str
    documents: List[Document]
    generation: str
    file_path: str


# 3. EXTRACTION NODE
def extraction_node(state: GraphState) -> Dict[str, Any]:
    file_path = state.get("file_path")
    if not file_path:
        return {"documents": []}

    print(f"\n[NODE: EXTRACTION] Processing file: {file_path}")
    extracted_docs = []

    try:
        if file_path.endswith(".pdf"):
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(file_path)
            extracted_docs = [Document(page_content=str(e)) for e in elements]
        elif file_path.lower().endswith((".png", ".jpg")):
            # NOTE: If you still need vision, you must keep Ollama/LLaVA
            # OR pass the image to a multimodal API like GPT-4o or Claude 3.5.
            # DeepSeek-V4-Pro is a text model.
            extracted_docs = [Document(page_content="Image processing requires a multimodal API.")]
    except Exception as e:
        print(f" -> Extraction failed: {e}")

    return {"documents": extracted_docs}


# 4. INTENT ROUTER
def intent_router(state: GraphState) -> Literal["chitchat", "search", "extraction"]:
    if state.get("file_path"):
        return "extraction"
    query = state["query"].lower().strip()
    conversational_triggers = ["hi", "hello", "hey", "thanks", "who are you", "what can you do", "help"]
    if any(trigger in query for trigger in conversational_triggers) or len(query.split()) <= 2:
        return "chitchat"
    return "search"


# 5. RETRIEVAL NODE
def retrieve_documents_node(state: GraphState) -> Dict[str, Any]:
    user_query = state["query"]
    retrieved_chunks = vector_store.similarity_search(user_query, k=5)
    combined_docs = state.get("documents", []) + retrieved_chunks
    return {"documents": combined_docs}


# 6. RERANK NODE
def rerank_documents_node(state: GraphState) -> Dict[str, Any]:
    user_query = state["query"]
    raw_docs = state["documents"]
    if not raw_docs: return {"documents": raw_docs}
    pairs = [[user_query, doc.page_content] for doc in raw_docs]
    scores = rerank_model.predict(pairs)
    scored_docs = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)
    filtered_docs = [doc for doc, score in scored_docs][:2]
    return {"documents": filtered_docs}


# 7. CORRECTIVE RAG NODE
def corrective_rag_node(state: GraphState) -> Dict[str, Any]:
    current_docs = state["documents"]
    is_relevant = any(
        term in doc.page_content.lower() for doc in current_docs for term in state["query"].split() if len(term) > 3)
    if is_relevant:
        return {"documents": current_docs}
    else:
        search_raw_output = web_search_tool(state["query"])
        fallback_doc = Document(page_content=search_raw_output, metadata={"source": "live_web_search_fallback"})
        return {"documents": [fallback_doc]}


# 8. GENERATION NODE
def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    user_query = state["query"]
    context_docs = state["documents"]
    formatted_context = "\n\n".join([doc.page_content for doc in context_docs])

    rag_prompt = f"CONTEXT LAYER: {formatted_context}\nUSER QUESTION: {user_query}\nCOMPUTED RESPONSE:"

    # Using your new service instead of 'llm.invoke'
    response = call_deepseek(rag_prompt)
    return {"generation": response}