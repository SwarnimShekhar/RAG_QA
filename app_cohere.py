import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
from dotenv import load_dotenv
import cohere
import time
from groq import Groq

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')

# Initialize LLMs
llm_local = ChatOllama(model="mistral:instruct")
llm_groq = Groq(api_key=groq_api_key)

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Streamlit UI setup
st.title("Document-Based QA with Pinecone , Groq & Cohere")
st.subheader("Upload a PDF, ask questions, and get accurate responses.")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Read and extract text from PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    # Split the PDF text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Embed documents using Cohere
    embeddings = CohereEmbeddings(model="embed-english-v3.0").embed_documents(texts)

    # List all available indexes
    existing_indexes = pinecone.list_indexes()

    # Check if the index exists
    if 'rag-qa-cohere' not in existing_indexes:
        # Create the index if it doesn't exist
        with st.spinner("Creating Pinecone index..."):
            pinecone.create_index(
                name='rag-qa-cohere',
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            st.write("Index 'rag-qa-cohere' created.")
    else:
        st.write("Index 'rag-qa-cohere' already exists.")

    # Connect to the existing index
    index = pinecone.Index("rag-qa-cohere")

    # Check if documents are already upserted to avoid re-upserting
    index_stats = index.describe_index_stats()
    if index_stats["total_vector_count"] == 0:
        with st.spinner("Upserting documents into Pinecone..."):
            for i in range(len(texts)):
                index.upsert([(str(i), embeddings[i], {"text": texts[i]})])
            st.success("Documents upserted successfully!")
    else:
        st.write("Documents already upserted to Pinecone.")

# Query Input
query = st.text_input("Enter your question:")
if query:
    # Get the query embedding using Cohere
    query_embedding = CohereEmbeddings(model="embed-english-v3.0").embed_query(query)

    # Rerank documents based on the query
    with st.spinner("Reranking relevant documents..."):
        query_result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        docs = {x["metadata"]['text']: i for i, x in enumerate(query_result["matches"])}
        co = cohere.Client(cohere_api_key)
        rerank_docs = co.rerank(model="rerank-english-v3.0", query=query, documents=list(docs.keys()), top_n=5, return_documents=True)
        context = [doc.document.text for doc in rerank_docs.results]

    # Generate response using Groq
    with st.spinner("Generating response..."):
        filled_template = f"Given the following context: {context}, generate a comprehensive and accurate response to the question: {query}. The response should include both paragraphs and bullet points where appropriate."
        chat_completion = llm_groq.chat.completions.create(
            messages=[{"role": "user", "content": filled_template}],
            model="mixtral-8x7b-32768",
        )
        response = chat_completion.choices[0].message.content
        st.write(response)

    # Display similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(context):
            st.write(f"Document {i + 1}:")
            st.write(doc)
            st.write("--------------------------------")

# Clear Pinecone index
if st.button("Clear Pinecone Index"):
    pinecone.delete_index("rag-qa-cohere")
    st.success("Pinecone index cleared successfully!")
