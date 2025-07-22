import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import run_conversation

# Initialize the messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Handle user input
def handle_user_input():
    user_input = st.session_state.user_input
    if user_input and user_input != "":
        st.session_state.messages.append({"role": "user", "content": user_input})
        langchain_messages = []
        
        # Convert the messages to LangChain message objects
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        # Run the conversation
        response = run_conversation(langchain_messages)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

# Display the messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

st.chat_input("What are you looking for?", key="user_input", on_submit=handle_user_input)
