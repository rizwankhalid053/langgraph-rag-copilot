import streamlit as st
import sys

# Import your compiled LangGraph application from main.py
from main import copilot_app

# --- WEB PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Corporate Knowledge Copilot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Corporate Knowledge Copilot")
st.caption("Powered by LangGraph, ChromaDB, & Llama 3.2")
st.markdown("---")

# --- INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I am your Enterprise Copilot assistant. Ask me anything about internal policies, project details, or external market trends."}
    ]

# --- DISPLAY EXISTING CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- USER PROMPT INPUT GATER ---
if user_prompt := st.chat_input("Ask a question..."):

    # 1. Display user message in chat UI container
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # 2. Generate Assistant Response with a loading spinner
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        with st.spinner("Propagating tokens through State Graph Matrix..."):
            try:
                # Prepare payload dictionary for LangGraph State
                input_payload = {
                    "query": user_prompt,
                    "documents": [],
                    "generation": ""
                }

                # Invoke our graph architecture synchronously
                execution_output = copilot_app.invoke(input_payload)
                final_answer = execution_output.get("generation", "Error: No response generated.")

                # Render answer on screen
                response_placeholder.write(final_answer)

                # Append to persistent session memory
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                error_msg = f"Runtime Pipeline Exception: {e}"
                response_placeholder.error(error_msg)