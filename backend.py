from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools.brave_search.tool import BraveSearch, BraveSearchWrapper
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

# Define the baweb search tool
@tool("web_search_tool")
def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web using Brave Search API
    Use this tool when you need to search the web with basic parameters.
    Answer in the same language as the one used by the human.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as formatted string
    """
    search = BraveSearch.from_api_key(
        api_key=BRAVE_SEARCH_API_KEY, 
        search_kwargs={"count": min(max_results, 20)}
    )
 
    result = search.run(query)
    print("-" * 80)
    print("Basic web search result:")
    print("-" * 80)
    pprint(result)
    return result

# Define the advanced web search tool
@tool("advanced_web_search")
def advanced_web_search(query: str, max_results: int = 5, country: str = "US", language: str = "en") -> str:
    """
    This tool searches the web with specific parameters.
    Use this tool when you need to search the web in a specific city, country and/or language.
    For example: 
     * use this tool to search in Paris, France, and search in French.
     * use this tool to search in Paris, France, and search in English.
     * use this tool to search in Paris, France, and search in Spanish.
     * use this tool to search in Paris, France, and search in German.
     * use this tool to search in Paris, France, and search in Italian.
     * use this tool to search in Paris, France, and search in Portuguese.
     * use this tool to search in Paris, France, and search in Russian.
    
    Args:
        query: Search query
        max_results: Number of results to return (1-20)
        country: Country code for localized results
        language: Language code for localized results

    Returns:
        Formatted search results
    """
    
    # Create wrapper with custom parameters
    search_wrapper = BraveSearchWrapper(
        api_key=BRAVE_SEARCH_API_KEY,
        search_kwargs={
            "count": min(max_results, 20),
            "country": country,
            "search_lang": language
        }
    )
    
    result = search_wrapper.run(query)
    print("-" * 80)
    pprint(search_wrapper.search_kwargs)
    print("Advanced web search result:")
    print("-" * 80)
    pprint(result)
    return result


# Define the tools
tools = [web_search_tool, advanced_web_search]
tool_node = ToolNode(tools=tools)

# Define the LLM with tools
#llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# Define the LLM node
def llm_node(state: MessagesState) -> MessagesState:
    response = llm_with_tools.invoke(state["messages"])
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