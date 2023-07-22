import os
from dotenv import load_dotenv
# from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
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

### clear all vectors from pinecone index
# index.delete(delete_all=True)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

# Get a list of all the files in ./documents
files = os.listdir('./documents')

# Loop over the files
for file in files:
    # Read the file in chunks and store them in a list
    chunks = []
    with open('./documents/' + file) as f:
        chunk_id = 0
        while True:
            chunk = f.read(500) # Read 500 characters at a time
            if not chunk:
                break
            # Add the chunk ID and text to the list
            chunks.append((chunk_id, chunk))
            chunk_id += 1

    # Batch insert the chunks into the vector store
    batch_size = 3 # Define your preferred batch size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        metadatas = [{'title': file,'text': chunk[1]} for chunk in batch]
        documents = [chunk for chunk_id, chunk in batch]
        embeds = embed.embed_documents(documents)
        ids = [str(chunk_id) for chunk_id, chunk in batch]
        index.upsert(vectors=zip(ids, embeds, metadatas))
        
    # # Flush the vector store to ensure all documents are inserted
    # vectorstore.flush()
    
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

# Dictionary to store chat IDs and their corresponding agents
chat_agents_dict = {}


class Query(BaseModel):
    chat_id: str
    query: str

@app.post("/ask")
async def ask(query: Query):
    # If chat_id is not in memory, create a new agent for it
    if query.chat_id not in chat_agents_dict:
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=8,
            return_messages=True
        )

        chat_agents_dict[query.chat_id] = initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=llm,
            verbose=False,
            max_iterations=3,
            early_stopping_method='generate',
            memory=conversational_memory
        )

    # Retrieve the correct agent for the chat_id
    agent = chat_agents_dict[query.chat_id]

    # Run the agent
    response = agent(query.query)
    output = response.get('output')
    
    ## get the relevant text 
    res = vectorstore.similarity_search(
        query=query.query,
        k=3  # return 3 most relevant docs
    )
    embedded_query = embed.embed_query(query.query)
    res2 = index.query(queries=[embedded_query], top_k=3)
    print(res2)

    def iter_output():
        out = ""
        for line in output.splitlines():
            out += line + "\n"
        page_contents = [res[0].page_content, res[1].page_content, res[2].page_content]
        meta_datas = [res[0].metadata, res[1].metadata, res[2].metadata]
        combined_text = [page_contents, meta_datas]
        
        ## crteate a readable text
        relevant_section = "\nRelevant Sections:\n"
        for i in range(0, len(combined_text)):
            relevant_section += str(i+1) + ". " + combined_text[0][i] + " :from:  " + combined_text[1][i]['title'] + "\n"
        out += relevant_section
        yield out
            

    return StreamingResponse(iter_output(), media_type="text/plain")
