import sys
from langgraph.graph import StateGraph, START, END

# Import shared state structure, configuration layout, and nodes
from nodes import GraphState, intent_router, retrieve_documents_node, rerank_documents_node, corrective_rag_node, \
    generate_answer_node

print("Assembling Agentic Workflow State Graph Topology...")

# 1. Initialize the State Graph workflow using our central notebook structural state
workflow = StateGraph(GraphState)

# 2. Add our execution workers (Nodes) to the graph matrix
workflow.add_node("retrieve_documents", retrieve_documents_node)
workflow.add_node("rerank_documents", rerank_documents_node)
workflow.add_node("corrective_rag_evaluator", corrective_rag_node)
workflow.add_node("generate_answer", generate_answer_node)

# 3. Define our Routing Entry Point 
# START will check the intent_router first. If chitchat, bypass database and jump straight to generation.
workflow.add_conditional_edges(
    START,
    intent_router,
    {
        "chitchat": "generate_answer",
        "search": "retrieve_documents"
    }
)

# 4. Define the sequential pipeline flow tracks
workflow.add_edge("retrieve_documents", "rerank_documents")
workflow.add_edge("rerank_documents", "corrective_rag_evaluator")
workflow.add_edge("corrective_rag_evaluator", "generate_answer")

# 5. Define the ending track boundary
workflow.add_edge("generate_answer", END)

# 6. Compile the graph into an executable application entity
copilot_app = workflow.compile()

print("--- AUTONOMOUS KNOWLEDGE COPILOT APPARATUS COMPILED SUCCESSFULLY ---\n")


def run_copilot_terminal_session():
    print("====================================================================")
    print("  Enterprise Knowledge Copilot Agent Online (LangGraph Architecture)")
    print("  Ask about internal policies, project details, or external market trends.")
    print("  Type 'exit' to gracefully end your interaction session.")
    print("====================================================================")

    while True:
        try:
            user_input = input("\nCopilot Prompt > ")
            if user_input.strip().lower() == 'exit':
                print("\nShutting down security contexts... Session ended safely. Goodbye!")
                sys.exit(0)

            if not user_input.strip():
                continue

            print("\n>>> Propagating token updates through State Graph Matrix...")

            # Formulate the payload dictionary that mirrors our GraphState definition
            input_payload = {"query": user_input, "documents": [], "generation": ""}

            # Invoke the graph synchronously
            execution_output = copilot_app.invoke(input_payload)

            # Print the final generated answer response cleanly on terminal console
            print("\n======================= FINAL RESPONSE OUTPUT =======================")
            print(execution_output.get("generation", "Error: No response generated."))
            print("=====================================================================")

        except KeyboardInterrupt:
            print("\n\nForced exit sequence initiated. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nRuntime Pipeline Exception: {e}")


if __name__ == "__main__":
    run_copilot_terminal_session()