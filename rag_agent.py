import os 

from dotenv import load_dotenv
from langchain import hub 
from langchain.agents import (
    AgentExecutor, 
    create_react_agent
)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.tools import Tool 
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import (
    AIMessage,
    HumanMessage
)

load_dotenv()

PERSISTENT_DIRECTORY = 'db'
REPO_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'

embedding_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

if os.path.exists(PERSISTENT_DIRECTORY):
    print("Loading from existing vector stores....")
    chroma_db = Chroma(persist_directory=PERSISTENT_DIRECTORY, embedding_function=embedding_model)

else:
    print("Creating new vector stores from the data books")
    BOOK_DIR = 'books'
    if os.path.exists(BOOK_DIR):
        books_file_paths = os.listdir(BOOK_DIR)[:6]
        documents = []
        
        # iterate through all files in `books` directory and create new vector stores based on the files
        for filepath in books_file_paths:
            loader = TextLoader(f'{BOOK_DIR}/{filepath}', encoding='utf-8')
            book_docs = loader.load()

            for doc in book_docs:
                doc.metadata = { 'source': filepath }
                documents.append(doc)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # outputting some information
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")

        chroma_db = Chroma.from_documents(persist_directory=PERSISTENT_DIRECTORY, embedding=embedding_model, documents=docs)

        print('Finished creating the document embeddings and loaded into the vector stores')

    else:
        raise FileNotFoundError('`books` directory does not exists!')
    
retriever = chroma_db.as_retriever(search_type='mmr', search_kwargs={ 'k': 2, 'fetch_k': 20, 'lambda_mult': 0.5 })

model = HuggingFaceEndpoint(repo_id=REPO_ID, temperature=0.1)

contextualized_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualized_q_prompt = ChatPromptTemplate.from_messages([
    ('system', contextualized_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

# help LLM to reformulate question to retrieve documents based on chat history
history_aware_retriever = create_history_aware_retriever(model, retriever, contextualized_q_prompt)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ('system', qa_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
])

# create a chain that combine retrieved documents into the prompt for QnA
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

react_docstore_prompt = hub.pull("hwchase17/react")

tools = [
    Tool(
        name="Answer Question", 
        description="Useful for when you need to answer questions about the context",
        func=lambda input, **kwargs: rag_chain.invoke({ 'input': input, 'chat_history': kwargs.get('chat_history', []) })
    )
]

agent = create_react_agent(llm=model, tools=tools, prompt=react_docstore_prompt, stop_sequence=True)

executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

chat_history = []
while True:
    prompt = input("You: ")
    if prompt.lower() == 'exit':
        break 

    response = executor.invoke({ 'input': prompt, 'chat_history': chat_history })
    print('AI:', response['output'])

    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=response['output']))

