import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Equity Research AI", page_icon="📈")
st.title("📈 Equity Research AI")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar for session info
with st.sidebar:
    st.info(f"Session ID: {st.session_state.session_id}")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about revenue growth, risks, or financial trends...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        response = requests.post(API_URL, json={
            "message": user_input,
            "session_id": st.session_state.session_id
        }).json()
        
        with st.chat_message("assistant"):
            st.write(response["reply"])
            if response.get("sources"):
                st.caption(f"Sources: {', '.join(response['sources'])}")
        
        st.session_state.messages.append({"role": "assistant", "content": response["reply"]})
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")