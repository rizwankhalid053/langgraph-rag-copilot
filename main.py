import sys
from langgraph.graph import StateGraph, START, END

# Import shared state, nodes, and router
from nodes import (
    GraphState,
    intent_router,
    extraction_node,
    retrieve_documents_node,
    rerank_documents_node,
    corrective_rag_node,
    generate_answer_node
)

print("Assembling Agentic Workflow State Graph Topology...")

workflow = StateGraph(GraphState)

# Register all nodes
workflow.add_node("extraction", extraction_node)
workflow.add_node("retrieve_documents", retrieve_documents_node)
workflow.add_node("rerank_documents", rerank_documents_node)
workflow.add_node("corrective_rag_evaluator", corrective_rag_node)
workflow.add_node("generate_answer", generate_answer_node)

# Routing Entry Point
workflow.add_conditional_edges(
    START,
    intent_router,
    {
        "chitchat": "generate_answer",
        "search": "retrieve_documents",
        "extraction": "extraction"
    }
)

# Pipeline flow
workflow.add_edge("extraction", "retrieve_documents")
workflow.add_edge("retrieve_documents", "rerank_documents")
workflow.add_edge("rerank_documents", "corrective_rag_evaluator")
workflow.add_edge("corrective_rag_evaluator", "generate_answer")
workflow.add_edge("generate_answer", END)

copilot_app = workflow.compile()
print("--- AUTONOMOUS KNOWLEDGE COPILOT APPARATUS COMPILED SUCCESSFULLY ---\n")

def run_copilot_terminal_session():
    # ... (Keep your existing terminal loop logic here)
    pass

if __name__ == "__main__":
    run_copilot_terminal_session()