from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools.brave_search.tool import BraveSearch, BraveSearchWrapper
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_exa import ExaSearchResults
from typing import List
from dotenv import load_dotenv, find_dotenv
import os
from pprint import pprint


# Load the environment variables
load_dotenv(find_dotenv(), override=True)

# Get the API key
#GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
#if GROQ_API_KEY is None:
#    raise ValueError("GROQ_API_KEY is not set")

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
print("MISTRAL_API_KEY", MISTRAL_API_KEY)
if MISTRAL_API_KEY is None:
    raise ValueError("MISTRAL_API_KEY is not set")

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# print("OPENAI_API_KEY", OPENAI_API_KEY)
# if OPENAI_API_KEY is None:
#     raise ValueError("OPENAI_API_KEY is not set")

# Get the Exa.ai API key
EXA_API_KEY = os.environ.get("EXA_API_KEY")
print("EXA_API_KEY", EXA_API_KEY)
if EXA_API_KEY is None:
    raise ValueError("EXA_API_KEY is not set")

# Define the exa.ai search tool
@tool("web_search_tool")
def web_search_tool(query: str, max_results: int = 5, user_location: str = "Belgium") -> str:
    """
    Search the web using exa.ai API.
    Use this tool when you need to search the web
    Answer in the same language as the one used by the human.

    Only search local business websites working in the construction industry.
    Do NOT search for:
	•	Job offers (linkedin, indeed, etc.)
	•	Job applications (linkedin, indeed, etc.)
	•	Job descriptions (linkedin, indeed, etc.)
	•	Job interview listings (linkedin, indeed, etc.)
    •	Local business directory (like yellow pages, yelp, etc.)
    •	Review platforms (google, yelp, etc.)
    *   Rental companies (like rent event space, rent a car, rent a bike, etc.)

    Args:
        query: The search query string
        
    Returns:
        Search results as formatted string in the same language the user used
    """
    search = ExaSearchResults(
        api_key=EXA_API_KEY, 
        text = True,
        type = "auto",
        category = "company",
        user_location = user_location,
        num_results = max_results
    )
 
    result = search.invoke(query)
    print("-" * 80)
    print("exa.ai search result:")
    print("-" * 80)
    pprint(result)
    return result


# Define the tools
tools = [web_search_tool]
tool_node = ToolNode(tools=tools)

# Define the LLM with tools
#llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
#llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
llm = ChatMistralAI(model="mistral-medium-latest", api_key=MISTRAL_API_KEY)
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

memory = MemorySaver()

# Compile the graph
graph = builder.compile(checkpointer=memory)

# LLM Judge to validate system prompt compliance
def llm_judge(response_content: str, user_query: str) -> dict:
    """
    Judge if the LLM response follows the system prompt rules.
    
    Args:
        response_content: The assistant's response content
        user_query: The original user query
        
    Returns:
        dict: {"compliant": bool, "violations": List[str], "score": float}
    """
    # Prefer OpenAI for judging when available, otherwise fall back to Mistral
    # if OPENAI_API_KEY:
    #     judge_llm = ChatOpenAI(model="gpt-5-mini", api_key=OPENAI_API_KEY, temperature=0)
    # else:
    #     judge_llm = ChatMistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY, temperature=0)
    judge_llm = ChatMistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY, temperature=0)
    judge_prompt = f"""
You are a compliance judge for a procurement assistant. Evaluate if the assistant's response follows these strict rules:

SYSTEM RULES:
1. Target only companies working in the construction industry
2. Use advanced search for specific city/country/language queries
3. Companies must operate in the country where construction project is located
4. Only search local business websites
5. Do NOT search for: job offers, job applications, job descriptions, job interview listings, business directories, review platforms
6. When searching in non-English speaking countries, search in local language but respond in the language of user_query
7. For each company: provide name and clickable URL to official website
8. Limit to 5 results

USER QUERY: "{user_query}"

ASSISTANT RESPONSE: "{response_content}"

Evaluate compliance and provide:
1. "compliant": true/false (overall compliance)
2. "violations": list of specific rule violations (empty if compliant)
3. "score": 0.0-1.0 (compliance score)
4. "explanation": brief explanation of assessment

Respond only with a JSON object.
"""

    try:
        judge_response = judge_llm.invoke([HumanMessage(content=judge_prompt)])
        import json
        
        # Extract JSON from response, handling potential markdown formatting
        response_text = judge_response.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(response_text)
        return result
    except Exception as e:
        print(f"Judge evaluation failed: {e}")
        print(f"Raw response: {judge_response.content if 'judge_response' in locals() else 'No response'}")
        return {"compliant": True, "violations": [], "score": 1.0, "explanation": "Judge evaluation failed"}

# Run the conversation with thread_id and compliance checking
def run_conversation(messages: List[BaseMessage], thread_id: str) -> BaseMessage:
    """Run the conversation through the graph and return the latest AI message."""
    print("Running conversation with thread_id:", thread_id)
    print("Messages:", messages)
    print("-" * 80)
    response = graph.invoke({"messages": messages}, config={"configurable": {"thread_id": thread_id}})
    print("Response:", response)
    print("-" * 80)

    # Get the last message from the response
    last_message = response["messages"][-1]
    
    # Extract user query (last human message)
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    # Judge the response for compliance
    if user_query and hasattr(last_message, 'content'):
        judgment = llm_judge(last_message.content, user_query)
        
        print("-" * 80)
        print("COMPLIANCE JUDGMENT:")
        print("-" * 80)
        pprint(judgment)
        
        # If not compliant, add a warning or modify response
        if not judgment.get("compliant", True):
            violations = judgment.get("violations", [])
            violation_text = "\n".join([f"• {v}" for v in violations])
            
            warning_message = f"""
⚠️ **Compliance Warning**: This response may not fully comply with system guidelines.

Detected violations:
{violation_text}

Score: {judgment.get('score', 0):.2f}/1.00
"""
            
            # Create a new message with the warning prepended
            modified_content = warning_message + "\n\n" + last_message.content
            last_message.content = modified_content
    
    # Ensure we return a proper message with content
    return last_message 
