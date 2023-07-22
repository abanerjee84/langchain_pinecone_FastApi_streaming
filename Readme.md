# Langchain Pinecone Chatbot

This is a simple API that uses Langchain, OpenAI, and Pinecone to create a conversational chatbot.Place your .txt files in "documents" folder.

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
- pandas
- tqdm

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

curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"chat_id":"3", "query":"Who was the bad boy?"}' \
  http://localhost:8000/ask


The context does not provide any specific information about who the "bad boy" referred to as Johny is. Therefore, we do not know who the bad boy is.

Relevant Sections:
1. nces, shaping the course of history and laying the foundation for the modern world order.


Johny was a bad boy.
 :from:  ww2.txt
2. th Axis and Allied forces, leading to the death of tens of millions of civilians through genocides, massacres, and starvation.

The aftermath of World War II saw the rise of two superpowers, the Soviet Union and the United States, and the onset of the Cold War, characterized by espionage, political subversion, and proxy wars. Europe was divided into the US-led Western Bloc and the USSR-led Eastern Bloc, leading to significant geopolitical tensions.
Overall, World War II had far-reaching conseque :from:  ww2.txt



curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"chat_id":"1", "query":"Who was the last president?"}' \
  http://localhost:8000/ask
The last president of the United States is Joe Biden.

Relevant Sections:
1. nces, shaping the course of history and laying the foundation for the modern world order.


Johny was a bad boy.
 :from:  ww2.txt
2. th Axis and Allied forces, leading to the death of tens of millions of civilians through genocides, massacres, and starvation.

The aftermath of World War II saw the rise of two superpowers, the Soviet Union and the United States, and the onset of the Cold War, characterized by espionage, political subversion, and proxy wars. Europe was divided into the US-led Western Bloc and the USSR-led Eastern Bloc, leading to significant geopolitical tensions.
Overall, World War II had far-reaching conseque :from:  ww2.txt

