from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import PromptTemplate

# --- Define a custom tool ---
@tool
def multiply_by_two(number: int) -> int:
    """Multiplies the input number by 2."""
    return number * 2

tools = [multiply_by_two]

# --- Load LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Define ReAct-style prompt directly ---
react_prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:"""
)

# --- Create ReAct Agent ---
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

# --- Wrap in Executor ---
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Run agent ---
response = agent_executor.invoke({"input": "What is 7 multiplied by 2, then add 5?"})
print("Final Answer:", response["output"])