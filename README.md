# Hotel assistent agent

Simple [LangGraph-based](https://www.langchain.com/langgraph) ReAct assistant agent for imaginary Hotel Monterey, located in the heart of Barcelona.


### Features
Displays hotel info (check-in/out times, restaurant schedule, Wi-Fi)
Utilizes [OpenWeather API](https://openweathermap.org/) for accurate weather information.

Hotel details were embedded in the system prompt to ensure they were readily available to the LLM, simplifying the architecture.


### Agent Decision Flow
```
[ User Query ]
       ↓
 [ ReAct Agent (LLM with system prompt) ]
       ↓
[ Decision: Tool Needed? ] ── No ──▶ [ Final Response (w/ prompt knowledge) ]
       │
      Yes
       ↓
 [ Tool Call (e.g., OpenWeather API) ]
       ↓
 [ Process Tool Output ]
       ↓
 [ Final Response (based on tool + prompt) ]
```

### Setting up the Environment

This project requires Python 3.9 or higher to run.

1) Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:
```
pip install -r requirements.txt
```

3) Create `.env` file and set environment variables:
```
OPENWEATHER_API_KEY=your_openweather_api_key
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_langsmith_project_name

# for LangSmith dataset export
DATASET_ID=...
```

### Running the Application
Run the agent script:
```
python agent.py
```

### Examples of user queries
```
What is the check-in time in the hotel?
What is the wifi credentials?
Weather in Barcelona today?
```
Press `q` to exit the chatbot.

### Evaluation experiment
The `dataset.py` contains an example of golden dataset test cases for evaluating generated responces. First create a dataset in LangSmith UI, set the `DATASET_ID` env variable and run using:
```
python dataset.py
```
Segment your data with [Dataset splits](https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application#create-and-manage-dataset-splits) on `full` contains diffirent examples and `critical`, contains only golden crucial examples.

Next setup [LLM-as-a-judge](https://docs.smith.langchain.com/evaluation/concepts#llm-as-judge) offline correctness [evaluator](https://docs.smith.langchain.com/evaluation/how_to_guides) using LangSmith UI to evaluate if the answer semantically matches a reference answer.

File `eval.py` contains LLM-as-a-judge [semantic similarity evaluator](https://github.com/langchain-ai/intro-to-langsmith/blob/main/notebooks/module_2/evaluators.ipynb) to run offline using the LangSmith SDK. We can use `Full` or `Critical` datasets for evaluation:
```
python eval.py
```
the output would be like:
```
View the evaluation results for experiment: 'gpt-4o-mini-407c7db1' at:
https://smith.langchain.com/o/d2dcad18-8f33/datasets/9067b5f5-ca2b/compare?selectedSessions=c7d66ab0
```

### Possible improvements
* Tools, heuristic and end-to-end performance evaluator experiments.
* More polite reply on out of scope questions rather than "I don't know.".
* Add vector store (see [RAG From Scratch series](https://github.com/langchain-ai/rag-from-scratch)) if there will be more custom data to ingest and expose, the flow might look like:
```
[ User Query ]
       ↓
 [ ReAct Agent (LLM with system prompt) ]
       ↓
[ Decision: Use Tool or RAG? ]
      ├── No ──▶ [ Final Response (w/ prompt knowledge) ]
      ├── Tool ──▶ [ Tool Call (e.g., OpenWeather API) ]
      │               ↓
      │         [ Process Tool Output ]
      │               ↓
      │       [ Final Response (tool + prompt) ]
      │
      └── RAG ──▶ [ Embed Query ]
                      ↓
             [ Search Vector Store ]
                      ↓
             [ Retrieve Context Chunks ]
                      ↓
             [ Grade + Select Relevant Info ]
                      ↓
             [ Final Response (retrieved + prompt) ]
```
* Add human-in-the-loop to the chatbot to handle situations where it may need guidance or verification before proceeding.
* Add memory e.g. short-term message history that keep track of previous conversations and context which can be included in the final prompt, as LLM's are stateless in nature.
* Add user feedback.

### Links
- 🔧 [LangGraph Docs](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- 🔧 [ReAct agent](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html)
- 📘 [LangSmith Evaluation Docs](https://docs.smith.langchain.com/evaluation)
- 🎓 [LangSmith Academy](https://academy.langchain.com/courses/intro-to-langsmith)