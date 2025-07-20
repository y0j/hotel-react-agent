import os
import dotenv
import requests
from datetime import datetime

dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from prompts import SYSTEM_PROMPT

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


llm = init_chat_model("openai:gpt-4o-mini", temperature=0)


@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # or 'imperial' for Fahrenheit
    }

    response = requests.get(url, params=params)
    data = response.json()

    return f"Current weather in {city}: {data['name']}, {data['main']['temp']}°C, condition:{data['weather'][0]['description']}"


# Create tools list
tools = [get_weather]

# Create the react agent with tools
agent = create_react_agent(llm, tools)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    # Add system prompt to the beginning of the conversation if not already present
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        system_message = SystemMessage(
            content=f"You are Hotel Monterey assistant. {SYSTEM_PROMPT}"
        )
        messages = [system_message] + messages

    # Use the react agent to process the messages
    result = agent.invoke({"messages": messages})

    # Return only the new assistant messages
    new_messages = result["messages"][len(messages) :]
    return {"messages": new_messages}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()


def stream_graph_updates(user_input: str):
    user_message = HumanMessage(content=user_input)
    for event in graph.stream({"messages": [user_message]}):
        for value in event.values():
            if value and "messages" in value and value["messages"]:
                # Only print the final assistant message (not tool calls)
                for msg in value["messages"]:
                    if (
                        hasattr(msg, "content")
                        and msg.content
                        and isinstance(msg, AIMessage)
                    ):
                        # Skip tool calls and only show final responses
                        if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                            print("Assistant:", msg.content)


# Infinite chat loop
print(
    "Assistant: Hello! Welcome to Hotel Monterey in Barcelona. How can I assist you today? I'm happy to provide you with the current weather forecast if you need it."
)
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(f"An error occurred: {e}")
        # Fallback behavior
        print("Let me try with a default question...")
        user_input = "What are your check-in and check-out times?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
