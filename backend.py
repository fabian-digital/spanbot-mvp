from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools.brave_search.tool import BraveSearch
from langchain_openai import ChatOpenAI
from typing import List
from dotenv import load_dotenv
import os
from pprint import pprint


# Load the environment variables
load_dotenv()

# Get the API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY is not set")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set")

# Get the Brave Search API key
BRAVE_SEARCH_API_KEY = os.environ.get("BRAVE_SEARCH_API_KEY")
if BRAVE_SEARCH_API_KEY is None:
    raise ValueError("BRAVE_SEARCH_API_KEY is not set")

# Define the web search tool
@tool("web_search_tool")
def web_search_tool(query: str) -> str:
    """
    Search the web using Brave Search API
    
    Args:
        query: The search query string
        
    Returns:
        Search results as formatted string
    """
    search = BraveSearch.from_api_key(
        api_key=BRAVE_SEARCH_API_KEY, 
        search_kwargs={"count": 3}
    )
    return search.run(query)

# Define the tools
tools = [web_search_tool]
tool_node = ToolNode(tools=tools)

# Define the LLM with tools
#llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# Define the LLM node
def llm_node(state: MessagesState) -> MessagesState:
    response = llm_with_tools.invoke(state["messages"])
    pprint(response)
    return {"messages": [response]}

# Define the graph
builder = StateGraph(MessagesState)
builder.add_node("llm_node", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm_node")
builder.add_conditional_edges("llm_node", tools_condition)
builder.add_edge("tools", "llm_node")
builder.add_edge("llm_node", END)

# Compile the graph
graph = builder.compile()

# Run the conversation
def run_conversation(messages: List[BaseMessage]) -> BaseMessage:
    """Run the conversation through the graph and return the latest AI message."""
    response = graph.invoke({"messages": messages})
    return response["messages"][-1] 