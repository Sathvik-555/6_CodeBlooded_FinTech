import streamlit as st
import os
import warnings
from dotenv import load_dotenv
import faiss
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Initialize environment variables and suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Resource Allocator")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        color : black;
    }
    .stApp {
        background-color: rgba(0, 0, 0, 0.5);  /* Transparent overlay for contrast */
    }
    .header {
        color: #007BFF;
        font-size: 36px;
        text-align: center;
        margin-top: 20px;
    }
    .response-section {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        color: black;
    }
    .error {
        color: red;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title with robot emoji
st.markdown('<h1 class="header">AI Optimized Resource Allocator </h1>', unsafe_allow_html=True)



with st.spinner("Loading content from the predefined website..."):
    try:
        # Load content from the website
        loader_multiple_pages = WebBaseLoader(["https://www.nseindia.com/", "https://portal.tradebrains.in/"])
        loader_multiple_pages.requests_kwargs = {'verify':False}


        docs = loader_multiple_pages.load()
        
        # Check if documents are empty
        if not docs:
            st.error("Failed to retrieve content. Please check the URL or the website's availability.")
        else:
            # Split documents into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)

            # Initialize embeddings
            embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="https://6codebloodedfintech-fqfjw2bhkkfd2ygbrh3wj6.streamlit.app/")

            # Create FAISS index
            d = len(embeddings.embed_query("test query"))  # Determine embedding dimensionality
            index = faiss.IndexFlatL2(d)
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )

            # Add documents to the vector store
            vector_store.add_documents(documents=chunks)

            # Define retriever and prompt template
            retriever = vector_store
            prompt = ChatPromptTemplate.from_template("""
                You are an AI-powered Finance Assistant designed to help users optimize their financial assets and resources. 
                Your tone should be professional, concise, and easy to understand.
                
                Question: {question} 
                Context: {context} 
                Answer:
            """)

            # Define RAG chain
            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])

            model = ChatOllama(model="llama3.2", base_url="https://6codebloodedfintech-fqfjw2bhkkfd2ygbrh3wj6.streamlit.app/")
            rag_chain = (
                RunnableMap({
                    "context": lambda query: retriever.search(query=query, search_type='similarity', k =5),
                    "question": RunnablePassthrough()
                })
                | (lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
                | prompt
                | model
                | StrOutputParser()
            )

            # Notify the user of successful document ingestion
    
            
            # Input field for user questions
            question = st.text_input("Ask a question about the extracted content:", key="question_input")

            if question:
                with st.spinner("Processing your query..."):
                    try:
                        # Get the answer using the RAG chain
                        answer = rag_chain.invoke(question)
                        st.markdown(f'<div class="response-section">{answer}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error">Error: {e}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
