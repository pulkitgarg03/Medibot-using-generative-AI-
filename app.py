from flask import Flask, render_template, jsonify, request
from src.helper import get_embeddings_model
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from dotenv import load_dotenv
from dotenv import dotenv_values



config = dotenv_values(".env")
PINECONE_API_KEY = config["PINECONE_API_KEY"]
HF_TOKEN = config["HF_TOKEN"]
GROQ_API_KEY = config["GROQ_API_KEY"]

load_dotenv()

app = Flask(__name__)

embedding_model = get_embeddings_model()

index_name = "medibot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding= embedding_model
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4,
    max_tokens=500
)

# model = ChatHuggingFace(llm = llm)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods =["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8000, debug= True)