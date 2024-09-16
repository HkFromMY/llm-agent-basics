from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent
)
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain import hub 
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

def get_current_time(*args, **kwargs):
    import datetime 

    return datetime.datetime.now().strftime("%I:%M %p")

def search_wikipedia(query):
    """Search wikipedia and return results"""

    from wikipedia import summary 

    try:
        return summary(query, sentences=2)
    
    except:
        return "I couldn't find any information on that."

tools = [
    Tool(name="Time", description="Useful for when you want to know the current time", func=get_current_time),
    Tool(name="Wikipedia", description="Useful for when you want to search information for a topic", func=search_wikipedia)
]

prompt = hub.pull("hwchase17/structured-chat-agent")

model = HuggingFaceEndpoint(repo_id='meta-llama/Meta-Llama-3-8B-Instruct', temperature=0.1)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt, stop_sequence=True)

executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

while True:
    prompt = input("You: ")
    if prompt.lower() == 'exit':
        break 

    memory.chat_memory.add_message(HumanMessage(content=prompt))

    response = executor.invoke({ 'input': prompt })
    print('AI:', response['output'])

    memory.chat_memory.add_message(AIMessage(content=response['output']))

