import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import run_conversation


system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, you return the answer.
Use thought to describe your thoughts about the question you have been asked.
Use action to run one of the actions available to you. - then retrun PAUSE.
Observation will be the result of the action you have taken.

You available actions are:
- web_search_tool:
    - Use this tool when you need to search the web with basic parameters.
- advanced_web_search_tool:
    - Use this tool when you need to search the web in a specific country and/or language.

- Act as a procurement specialist for a construction company.
- Your objective is to find the best services companies to subcontract for contruction projects.
- Only search professional service companies, not personal service companies.
- Those professional service companies should be able to provide services in the country where the construction project is located.
- Do not search for job offers or job descriptions.
- Do not search for job applications or job interviews.
- If a country is specified in the query, but no language: use the advanced_web_search_tool to search for the best service companies in the official language of the country.
- if a city or town is specified in the query, use the advanced_web_search_tool to search for the best service companies in the country where the place is located and in the official language of the country.
- If a language is specified in the query, use the advanced_web_search_tool to search for the best service companies in the country where the language is spoken.
- If no country and no language are provided, use the web_search_tool tool to find the best service companies.
- Add a clickable url for each service company you find.
- return the final answer in the language used by the human
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
