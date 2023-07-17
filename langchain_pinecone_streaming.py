import os
from dotenv import load_dotenv
# from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain, ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
from langchain.agents import initialize_agent
import pinecone
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Any, Dict
from pydantic import BaseModel

# Load the environment variables
load_dotenv()

# first initialize the large language model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# find API key in console at app.pinecone.io
YOUR_API_KEY = os.getenv("PINECONE_API_KEY")
# find ENV (cloud region) next to API key in console
YOUR_ENV = os.getenv("PINECONE_ENVIRONMENT")
# find INDEX_NAME in console under "Indices" tab
YOUR_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)



model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)


index = pinecone.Index(YOUR_INDEX_NAME)
text_field = "text"

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=8,
    return_messages=True
)

# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query1 = 'Who is the president of the United States?'
# print(qa.run(query1))
query2 = 'how old is he?'
# print(qa.run(query2))

from langchain.agents import Tool

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=False,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)


app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/ask")
async def ask(query: Query):
    response = agent(query.query)
    output = response.get('output')

    def iter_output():
        for line in output.splitlines():
            yield line + "\n"

    return StreamingResponse(iter_output(), media_type="text/plain")
