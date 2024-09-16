from dotenv import load_dotenv
from langchain.tools import Tool
from langchain import hub 
from langchain.agents import (
    AgentExecutor,
    create_react_agent
)
from langchain_community.llms import HuggingFaceEndpoint

REPO_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'

load_dotenv()

def get_current_time(*args, **kwargs):
    # the params is set to avoid conflict with the model 

    import datetime 

    return datetime.datetime.now().strftime("%I:%M %p")

tools = [
    Tool(name="Time", description="Useful for when you want to know the current time", func=get_current_time)
]

prompt = hub.pull("hwchase17/react")

model = HuggingFaceEndpoint(repo_id=REPO_ID, temperature=0.1)

# create an agent that specifies which llm, prompt and tools to be used
agent = create_react_agent(llm=model, tools=tools, prompt=prompt, stop_sequence=True)

# executor that executes as that agent passed in, and the tool list to select from
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

response = executor.invoke({ 'input': 'What time is it now?' })

print('Response:', response)
