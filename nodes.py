from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Pull active, validated infrastructure configurations
from config import vector_store, rerank_model, web_search_tool, llm


class GraphState(TypedDict):
    query: str
    documents: List[Document]
    generation: str


def intent_router(state: GraphState) -> Literal["chitchat", "search"]:
    print("\n[ROUTER CONDITIONAL EDGE] Evaluating user conversational intent...")
    query = state["query"].lower().strip()

    # Catch conversational intents to instantly bypass storage engines
    conversational_triggers = ["hi", "hello", "hey", "thanks", "who are you", "what can you do", "help"]
    if any(trigger in query for trigger in conversational_triggers) or len(query.split()) <= 2:
        print(" -> Intent Classified: [CHITCHAT]. Bypassing storage pools.")
        return "chitchat"

    print(" -> Intent Classified: [SEARCH]. Activating enterprise data search track.")
    return "search"


def retrieve_documents_node(state: GraphState) -> Dict[str, Any]:
    print("\n[NODE: RETRIEVER] Querying Chroma Database...")
    user_query = state["query"]
    retrieved_chunks = vector_store.similarity_search(user_query, k=5)
    print(f" -> Collected {len(retrieved_chunks)} raw candidate text fragments from database.")
    return {"documents": retrieved_chunks}


def rerank_documents_node(state: GraphState) -> Dict[str, Any]:
    print("\n[NODE: RE-RANKER] Executing Cross-Encoder context filtering...")
    user_query = state["query"]
    raw_docs = state["documents"]

    if not raw_docs:
        return {"documents": raw_docs}

    pairs = [[user_query, doc.page_content] for doc in raw_docs]
    scores = rerank_model.predict(pairs)
    scored_docs = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)

    filtered_docs = [doc for doc, score in scored_docs][:2]
    print(f" -> Noise filtering complete. Retained top {len(filtered_docs)} high-relevance chunks.")
    return {"documents": filtered_docs}


def corrective_rag_node(state: GraphState) -> Dict[str, Any]:
    print("\n[NODE: CORRECTIVE RAG EVALUATOR] Checking data quality bounds...")
    user_query = state["query"].lower()
    current_docs = state["documents"]

    query_terms = [term for term in user_query.split() if len(term) > 3]
    is_relevant = False
    for doc in current_docs:
        content = doc.page_content.lower()
        if sum(1 for term in query_terms if term in content) >= 1:
            is_relevant = True
            break

    if is_relevant:
        print(" -> Evaluation Result: [RELEVANT]. Local context database payload approved.")
        return {"documents": current_docs}
    else:
        print(" -> Evaluation Result: [IRRELEVANT]. Internal data is missing context or outdated.")
        print(" -> CRAG Intervention: Launching live emergency external web search...")
        try:
            # Native functional invocation fix applied here
            search_raw_output = web_search_tool(state["query"])
            fallback_doc = Document(
                page_content=search_raw_output,
                metadata={"source": "live_web_search_fallback"}
            )
            print(" -> External search payload secured. Overwriting system context.")
            return {"documents": [fallback_doc]}
        except Exception as e:
            print(f" -> Fallback Search Error: {e}. Defaulting back to internal documents.")
            return {"documents": current_docs}


def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    print("\n[NODE: OLLAMA RESPONSE ENGINE] Synthesizing final response output...")
    user_query = state["query"]
    context_docs = state["documents"]

    formatted_context = "\n\n".join([doc.page_content for doc in context_docs])

    # Clean, balanced, contextual prompt engineered explicitly for Llama 3.2
    rag_template = """You are an intelligent, real-time Corporate Knowledge Copilot.
Your goal is to answer the user's question accurately using the information provided in the Context layer below.

CONTEXT LAYER (Live Web Snippets / Company Records):
{context}

USER QUESTION: 
{question}

COMPUTED RESPONSE:"""

    prompt_payload = PromptTemplate(template=rag_template, input_variables=["context", "question"])
    final_prompt = prompt_payload.format(
        context=formatted_context if formatted_context.strip() else "No real-time web or internal records could be retrieved for this query.",
        question=user_query
    )

    response = llm.invoke(final_prompt)
    return {"generation": response}