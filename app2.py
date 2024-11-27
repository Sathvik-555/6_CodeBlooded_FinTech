import streamlit as st
import os
import warnings
from dotenv import load_dotenv
import faiss
from langchain_community.document_loaders import PyMuPDFLoader
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

# Load and process PDFs (predefined, no user uploads)
pdf_directory = "C:/Users/Sathvik/OneDrive/Desktop/chatbot"
pdfs = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

if not pdfs:
    raise ValueError(f"No PDF files found in directory {pdf_directory}.")
docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    try:
        pages = loader.load()
        docs.extend(pages)
    except Exception as e:
        st.error(f"Error loading {pdf}: {e}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Initialize embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

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
prompt =ChatPromptTemplate.from_template("""
You are an AI-powered assistant designed exclusively to assist students of RVCE (Rashtreeya Vidyalaya College of Engineering) with college-related tasks. Your tone should be friendly, approachable, and professional. Provide accurate and concise answers, and always stay within the scope of college-related queries.

You can assist students in the following areas:
- **Academic help:** Information about classes, schedules, assignments, exams, and study tips.
- **Campus resources:** Guidance on accessing campus facilities like the library, labs, or counseling services.
- **Events and activities:** Details about clubs, events, and extracurricular activities.
- **Administrative help:** Queries about registration, fee payment, academic records, and other administrative tasks.
- **Personal organization:** Suggestions for time management, stress handling, and productivity tools.

**Guidelines for responses:**
1. If the question is unrelated to RVCE or the scope above, respond politely by stating that you only provide information about RVCE and its students.
2. Avoid making assumptions or providing information you are unsure about. Direct users to appropriate college resources or offices for further assistance.
3. Always use a conversational tone to make the interaction welcoming and helpful.

**Example interactions:**

**Student:** "How do I register for a workshop?"
**Chatbot:** "You can register for workshops through the RVCE student portal or by contacting the specific department organizing it. Let me know if you need more details!"

**Student:** "What is the population of India?"
**Chatbot:** "I’m here to assist with queries related to RVCE and its students. For general questions, you might want to try a search engine."

Question: {question} 
Context: {context} 
Answer:
""")


# Format documents for the model
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Initialize ChatOllama model
model = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

# Define RAG chain
rag_chain = (
    RunnableMap({
        "context": lambda query: retriever.search(query=query, search_type='similarity'),
        "question": RunnablePassthrough()
    })
    | (lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
    | prompt
    | model
    | StrOutputParser()
)


# Set page configuration
st.set_page_config(page_title="RVCE Chatbot", page_icon=":robot_face:")
st.image("C:/Users/Sathvik/Downloads/rvce.jpg", width=150)


# Custom CSS for setting the background image
st.markdown(f"""
    <style>
    body {{
        font-family: 'Arial', sans-serif;
        background-image: url("C:/Users/Sathvik/Downloads/bg.jpg")
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.5);  /* Transparent overlay for contrast */
    }}
    .header {{
        color: #007BFF;
        font-size: 36px;
        text-align: center;
        margin-top: 20px;
    }}
    .question-input {{
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
    }}
    .response-section {{
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
    }}
    .error {{
        color: red;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# Title with robot emoji next to it using HTML
st.markdown('<h1 class="header">RVCE Chatbot <span style="font-size: 40px;">🤖</span></h1>', unsafe_allow_html=True)

# Description
st.markdown('<h3>Ask questions about your syllabus, timetable, or anything related to your college experience.</h3>', unsafe_allow_html=True)

# Input field for user questions
question = st.text_input("Enter your question:", key="question_input", help="Ask about your classes, exams, or any other college-related queries.")

if question:
    with st.spinner("Processing your query..."):
        try:
            # Invoke the RAG chain to get the response (assuming rag_chain is defined)
            answer = rag_chain.invoke(question)
            
            st.write(answer)
        except Exception as e:
            st.markdown(f'<div class="error">Error: {e}</div>', unsafe_allow_html=True)