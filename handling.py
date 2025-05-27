import streamlit as st
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Coffee with ChatBot ☕", page_icon="🤖", layout="centered")

# Inject custom dark mode CSS
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .user-msg, .bot-msg {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 20px;
            max-width: 80%;
            word-wrap: break-word;
            font-size: 16px;
        }
        .user-msg {
            background-color: #3b5998;  /* Dark Blue */
            color: #ffffff;
            align-self: flex-end;
            margin-left: auto;
            text-align: right;
        }
        .bot-msg {
            background-color: #2c2f33;  /* Dark Gray */
            color: #ffffff;
            align-self: flex-start;
            margin-right: auto;
            text-align: left;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        .title-container {
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        .chat-box {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 15px;
            min-height: 200px;
            max-height: 600px;
            overflow-y: auto;
        }
        input, textarea {
            background-color: #2a2a2a !important;
            color: white !important;
        }
        .stTextInput>div>div>input {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-container"><h1>☕ Coffee with ChatBot 🤖</h1></div>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

# Define LLM and prompt
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
    MessagesPlaceholder(variable_name="history")
])

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything...", placeholder="Type your message here...")
    submit = st.form_submit_button("Send")

# Handle submission
if submit and user_input:
    st.session_state.history.add_user_message(user_input)

    chat_chain = prompt | llm
    chat_with_memory = RunnableWithMessageHistory(
        chat_chain,
        lambda session_id: st.session_state.history,
        input_messages_key="question",
        history_messages_key="history"
    )

    response = chat_with_memory.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": "chat123"}}
    )
    bot_reply = response.content

    # Update message history
    st.session_state.history.add_ai_message(bot_reply)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

# Display messages
with st.container():
    st.markdown('<div class="chat-box chat-container">', unsafe_allow_html=True)

    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="text-align:center; color:gray;">Start a conversation above 👆</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
