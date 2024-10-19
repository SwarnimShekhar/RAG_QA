import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
import cohere
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from the .env file
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')

# Initialize LLMs
llm_local = ChatOllama(model="mistral:instruct")
llm_groq = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

# Split the text into chunks
def split_text(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(pdf_text)

# Embed documents using Cohere
def embed_documents(texts):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    return embeddings.embed_documents(texts)

# Upsert into Pinecone
def upsert_documents(texts, embeddings):
    pc = Pinecone(api_key=pinecone_api_key)
    pc.create_index(name="rag-qa-cohere", dimension=1024, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    index = pc.Index("rag-qa-cohere")

    for i in range(len(texts)):
        index.upsert([(str(i), embeddings[i], {"text": texts[i]})])
    
    print("Done upserting...")
    return index

# Get query embedding using Cohere
def get_query_embedding(text):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    return embeddings.embed_query(text)

# Rerank documents
def rerank_documents(query, question_embedding, index):
    query_result = index.query(vector=question_embedding, top_k=5, include_metadata=True)
    docs = {x["metadata"]['text']: i for i, x in enumerate(query_result["matches"])}

    co = cohere.Client(cohere_api_key)
    rerank_docs = co.rerank(model="rerank-english-v3.0", query=query, documents=list(docs.keys()), top_n=5, return_documents=True)

    return [doc.document.text for doc in rerank_docs.results]

# Generate response
def generate_response(context, query):
    client = Groq(api_key=groq_api_key)
    filled_template = f"Given the following context: {context}, generate a comprehensive and accurate response to the question: {query}. The response should include both paragraphs and bullet points where appropriate."
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": filled_template}],
        model="mixtral-8x7b-32768",
    )
    
    return chat_completion.choices[0].message.content
