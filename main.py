from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from IPython.display import display, Markdown
from pydantic.types import T
from typing_extensions import TypedDict
from typing import List

import streamlit as st

from dotenv import load_dotenv
import os

# Check if the API key is set
if os.environ.get("GROQ_API_KEY") is None:
    load_dotenv()
    API_KEY = os.environ.get("GROQ_API_KEY")
    if API_KEY is None:
        raise ValueError("GROQ_API_KEY is not set")
else:
    API_KEY = os.environ.get("GROQ_API_KEY")


# Define the state
class State(TypedDict):
    messages: List[BaseMessage]

# Define the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)

# Define the LLM node
def llm_node(state: MessagesState) -> MessagesState:
    print(state)
    print("#" * 80)
    print("LLM NODE")
    print(state["messages"])

    # Simply invoke the LLM with the messages
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Define the graph
builder = StateGraph(MessagesState)
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

# Compile the graph
graph = builder.compile()

#display(Markdown(graph.get_graph().draw_mermaid_png()))

# Initialize the messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Callback function for user input
def handle_user_input():
    # Get the user input
    user_input = st.session_state.user_input
    if user_input and user_input != "":
        # Add the user input to the messages
        st.session_state.messages.append({"role": "user", "content": user_input})
 
        # Convert session state messages to LangChain message format
        langchain_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # Invoke the graph with the full conversation history
        response = graph.invoke({"messages": langchain_messages})
        
        # Add the response to the messages
        st.session_state.messages.append({"role": "assistant", "content": response["messages"][-1].content})
     
# Display the messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Add a chat input widget
st.chat_input("What are you looking for?", key="user_input", on_submit=handle_user_input)
