import streamlit as st

from langchain_huggingface import  ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
load_dotenv()


# Streamlit Page Config

st.set_page_config(page_title="Medical RAG Bot", page_icon="🩺")

st.title("🩺 Medical Chatbot (RAG + Sources)")
st.write("Ask medical questions based only on your knowledge base.")



# Initialize Session State

if "history" not in st.session_state:
    st.session_state.history = []



# Prompt Template

template = """
Use ONLY the information provided in the context to answer.

If you don't know the answer, say: "I don't know."

Context:
{context}

Chat History:
{history}

Question:
{question}

Answer directly:
"""


def set_prompt():
    return PromptTemplate(
        template=template,
        input_variables=["context", "question", "history"]
    )



# Load Model 

@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm)


model = load_model()



# Load FAISS Vectorstore (Cached)

@st.cache_resource
def load_vectorstore():
    db_faiss_path = "vectorstore/db_faiss"

    emb_model = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = FAISS.load_local(
        db_faiss_path,
        embeddings=emb_model,
        allow_dangerous_deserialization=True
    )
    return db


db = load_vectorstore()



# Retriever

retriever = db.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])



# ✅ Build RAG Chain (Answer Only)

rag_chain = (
    {
        "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        "question": RunnableLambda(lambda x: x["question"]),
        "history": RunnableLambda(lambda x: x["history"]),
    }
    | set_prompt()
    | model
    | StrOutputParser()
)



# Sidebar Controls

st.sidebar.title("Controls")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.history = []
    st.rerun()



# Display Chat Messages

for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)



# User Input Box

user_query = st.chat_input("Type your medical question...")


if user_query:

    # Show user message
    st.chat_message("user").write(user_query)
    st.session_state.history.append(HumanMessage(content=user_query))

    # Build history text
    history_text = "\n".join([m.content for m in st.session_state.history])

   
    # Retrieve Docs Separately (For Sources)

    retrieved_docs = retriever.invoke(user_query)

    # Generate Answer
    response = rag_chain.invoke(
        {
            "question": user_query,
            "history": history_text
        }
    )

    # Show assistant response
    st.chat_message("assistant").write(response)
    st.session_state.history.append(AIMessage(content=response))

   
    # Show Sources
  
    st.markdown("### 📌 Sources Used:")

    for i, doc in enumerate(retrieved_docs, start=1):

        source_name = doc.metadata.get("source", "Unknown Source")

        with st.expander(f"Source {i}: {source_name}"):
            st.write(doc.page_content[:500] + "...")
