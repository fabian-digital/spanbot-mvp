import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import run_conversation

# Configure the page
st.set_page_config(
    page_title="SpanTech Procurement Assistant",
    page_icon="🏗️",
    layout="wide"
)

# Add sidebar with logo
with st.sidebar:
    st.image("static/img/spantech_logo.png", width=200)
    st.markdown("---")
    st.markdown("### Procurement Assistant")
    st.markdown("Find the best professional service companies for your construction projects.")


system_prompt = """
Procurement Analyst Agent for Construction Projects

You are a procurement specialist for a construction company. Your job is to identify and recommend the best professional service companies to subcontract for construction projects.

Follow these strict guidelines:
	1.	Target only professional service companies in the construction industry.
    2.  You must use the advanced search tool to search for companies in a specific city, country and/or language.
	3.  The companies must operate in the country where the construction project is located.
	4..	Do not search for:
	•	Job offers
	•	Job applications
	•	Job descriptions
	•	Job interview listings
	5.	When the user requires searching in a city within a country that speaks a language other than English, you must use the advanced search tool.
        Search in the local language of the country. For example:
        - If the user is searching in a city in Spain, search in Spanish.
        - If the user is searching in a city in France, search in French.
        - If the user is searching in a city in Germany, search in German.
        - If the user is searching in a city in Italy, search in Italian.
        - If the user is searching in a city in Portugal, search in Portuguese.
        - If the user is searching in a city in Russia, search in Russian.
        - If the user is searching in a city in Spain, search in Spanish.
        - If the user is searching in a city in Switzerland, search in German.
        - If the user is searching in a city in Turkey, search in Turkish.
	6.	Despite searching in the local language, you must respond to the user in their own language (the language they used when asking the question).
	7.	For each recommended company:
	•	Provide the company name
	•	Include a clickable URL to their official website or listing
	8.  Return the final answer in the same language the user used.
"""

# Initialize the messages
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

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
