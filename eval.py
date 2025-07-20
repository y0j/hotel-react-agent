import os
import dotenv
from datetime import datetime
from prompts import SYSTEM_PROMPT

dotenv.load_dotenv()

from langsmith import traceable, wrappers
from openai import OpenAI
from pydantic import BaseModel, Field
from langsmith import evaluate
from langsmith import Client
from langsmith import traceable

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# Create LangSmith client
LangSmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Full and Critical datasets to evaluate against, we can choose one on runtime
full_dataset = LangSmith_client.list_examples(
    dataset_name="monterey-agent Golden Dataset"
)
critical_dataset = LangSmith_client.list_examples(
    dataset_name="monterey-agent Golden Dataset", splits=["Critical set"]
)


# Needed to run create_react_agent, mock function
@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    return f"Current weather in {city}: temp °C, condition: description"


# Create tools list
tools = [get_weather]

llm = init_chat_model("openai:gpt-4o-mini", temperature=0)

# Create the react agent with tools
agent = create_react_agent(llm, tools)


# Define the state for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define the chatbot function
@traceable(name="Extract Chatbot Details")
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


# Build the state graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()


# Stream updates from the graph
@traceable(name="Stream Graph Updates")
def stream_graph_updates(user_input: str) -> str:
    user_message = HumanMessage(content=user_input)
    final_response = None  # capture final assistant message

    for event in graph.stream({"messages": [user_message]}):
        for value in event.values():
            if value and "messages" in value and value["messages"]:
                for msg in value["messages"]:
                    if isinstance(msg, AIMessage):
                        if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                            final_response = msg.content
    return final_response


# Target function for LangSmith evaluation
@traceable(name="Target Function")
def target_function(inputs: dict) -> dict:
    user_input = inputs["HUMAN"]
    assistant_response = stream_graph_updates(user_input)
    return {"output": assistant_response}


# OpenAI client for evaluation
client = OpenAI()


# Define the Pydantic model for the similarity score
class Similarity_Score(BaseModel):
    similarity_score: int = Field(
        description="Semantic similarity score between 1 and 10, where 1 means unrelated and 10 means identical."
    )


# Evaluator function
def compare_semantic_similarity(inputs: dict, reference_outputs: dict, outputs: dict):
    input_question = inputs["HUMAN"]
    reference_response = reference_outputs["AI"]
    run_response = outputs["output"]

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a semantic similarity evaluator. Compare the meanings of two responses to a question, "
                    "Reference Response and New Response, where the reference is the correct answer, and we are trying to judge if the new response is similar. "
                    "Provide a score between 1 and 10, where 1 means completely unrelated, and 10 means identical in meaning."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {input_question}\n Reference Response: {reference_response}\n Run Response: {run_response}",
            },
        ],
        response_format=Similarity_Score,
    )

    similarity_score = completion.choices[0].message.parsed
    return {"score": similarity_score.similarity_score, "key": "similarity"}


# Run the evaluation
results = LangSmith_client.evaluate(
    target_function,
    data=critical_dataset,
    evaluators=[compare_semantic_similarity],
    experiment_prefix="gpt-4o-mini",
    description="Semantic similarity evaluation",
    max_concurrency=4,
)
