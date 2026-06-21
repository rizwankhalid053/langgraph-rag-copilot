import streamlit as st
import os
from main import copilot_app

st.set_page_config(
    page_title="Corporate Knowledge Copilot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Corporate Knowledge Copilot")
st.caption("Now supports PDF and Image (Chart) Extraction")
st.markdown("---")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload a PDF or Chart (PNG/JPG)", type=["pdf", "png", "jpg"])

# Use session state to persist the file path for the current "transaction"
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = ""

if uploaded_file:
    os.makedirs("./temp_uploads", exist_ok=True)
    temp_path = os.path.join("./temp_uploads", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.temp_file_path = temp_path
    st.success(f"File uploaded: {uploaded_file.name}")

# --- INITIALIZE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask about your documents or market trends!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- INPUT HANDLING ---
if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing through Agentic Graph..."):
            try:
                # Construct payload
                # We use the path if it exists, otherwise pass an empty string
                input_payload = {
                    "query": user_prompt,
                    "documents": [],
                    "generation": "",
                    "file_path": st.session_state.temp_file_path
                }

                execution_output = copilot_app.invoke(input_payload)
                final_answer = execution_output.get("generation", "Error generating response.")

                st.write(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

                # --- CRITICAL FIX: Clear file path after processing ---
                # This ensures the next question doesn't re-process the same PDF
                st.session_state.temp_file_path = ""

            except Exception as e:
                st.error(f"Runtime Pipeline Exception: {e}")