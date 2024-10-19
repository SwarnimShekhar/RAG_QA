import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings  # Updated import
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
from dotenv import load_dotenv
import cohere

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')

# Initialize LLMs
llm_local = ChatOllama(model="mistral:instruct")
llm_groq = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Streamlit UI setup
st.title("Document-Based QA with Pinecone, Groq & Cohere")
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

    # Embed documents
    embeddings = CohereEmbeddings(model="embed-english-v3.0")  # Updated class
    embeddings = embeddings.embed_documents(texts)

    # Check if the index already exists
    index_name = 'rag-qa-cohere'
    existing_indexes = pinecone.list_indexes()

    if index_name not in existing_indexes:
        # Create the index if it doesn't exist
        with st.spinner("Creating Pinecone index..."):
            pinecone.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            st.success(f"Index '{index_name}' created.")
    else:
        st.write(f"Index '{index_name}' already exists.")

    # Connect to the existing index
    index = pinecone.Index(index_name)

    # Check if documents are already upserted
    index_stats = index.describe_index_stats()

    if index_stats["vectorCount"] == 0:
        # If no vectors exist, proceed with upserting documents
        with st.spinner("Upserting documents into Pinecone..."):
            embeddings = CohereEmbeddings(model="embed-english-v3.0").embed_documents(texts)
            for i in range(len(texts)):
                index.upsert([(str(i), embeddings[i], {"text": texts[i]})])
            st.success("Documents upserted successfully!")
    else:
        st.write("Documents are already upserted to Pinecone.")


# Query Input
query = st.text_input("Enter your question:")
if query:
    # Get the query embedding
    query_embedding = CohereEmbeddings(model="embed-english-v3.0").embed_query(query)  # Updated class

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
