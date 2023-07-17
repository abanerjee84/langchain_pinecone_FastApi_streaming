# Langchain Pinecone Chatbot

This is a simple API that uses Langchain, OpenAI, and Pinecone to create a conversational chatbot.

## Setup

### Prerequisites

```bash
##### Ensure Python 3.6 or higher is installed.

# Install necessary Python packages:
pip install -r requirements.txt
## The requirements file should include:
- fastapi
- pinecone-client
- uvicorn
- python-dotenv
- langchain

# Set up environment variables:
### Create a `.env` file in your project root and set your credentials:
OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
PINECONE_API_KEY='YOUR_PINECONE_API_KEY'
PINECONE_ENVIRONMENT='YOUR_PINECONE_ENVIRONMENT'
PINECONE_INDEX_NAME='YOUR_PINECONE_INDEX_NAME'

Replace `YOUR_OPENAI_API_KEY`, `YOUR_PINECONE_API_KEY`, `YOUR_PINECONE_ENVIRONMENT`, and `YOUR_PINECONE_INDEX_NAME` with your actual OpenAI API Key, Pinecone API Key, Pinecone Environment, and Pinecone Index Name respectively.

# Running the API Server
# You can start the API server with the following command:
uvicorn langchain_pinecone_streaming:app --reload

The server will start on `http://localhost:8000`.

# USAGE
The API provides one endpoint: `POST /ask`
This endpoint expects a JSON object with a `query` field in the request body.
### Example:

curl -X POST "http://localhost:8000/ask" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"query\":\"Who is the president of the United States?\"}"

As of my last update in October 2021, the President of the United States is Joe Biden.

curl -X POST "http://localhost:8000/ask" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"query\":\"How old is he?\"}"

Joe Biden is 78 years old
