import streamlit as st
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os

# Set API key
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

st.title("📚 Context-Aware Chatbot with RAG (Groq + Memory)")

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})

retriever = load_retriever()
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer. Keep concise.\n\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": itemgetter("input") | retriever | format_docs,
        "chat_history": itemgetter("chat_history"),
        "input": itemgetter("input"),
    }
    | prompt
    | llm
)

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.chat_message("user").write(user_input)
    with st.spinner("Thinking..."):
        response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "streamlit_user"}}
        )
        answer = response.content
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})
