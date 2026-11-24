import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Adjust these import paths to match your actual folder structure.
from src.graph.graph_builder import GraphBuilder
from src.llm.groqllm import GroqLLM


# Load environment variables
load_dotenv()


# --------------------------------
# Streamlit Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Hi...., I'm Chiti",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------
# Initialize Chatbot
# --------------------------------

@st.cache_resource
def create_graph(model_name):

    user_input_control = {
        "selected_model": model_name
    }

    # Initialize Groq LLM
    groq_llm = GroqLLM(user_input_control)
    model = groq_llm.get_llm_model()

    # Build LangGraph
    graph_builder = GraphBuilder(model)
    graph = graph_builder.setup_graph("Basic Chatbot")

    return graph


# --------------------------------
# Session State
# --------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------
# Sidebar
# --------------------------------

with st.sidebar:

    st.title("🤖 AI Assistant")

    st.caption("Powered by LangGraph and Groq")

    st.divider()

    selected_model = st.selectbox(
        "Select LLM Model",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ]
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("### About")

    st.info(
        "This chatbot uses LangGraph to manage "
        "conversation flow and Groq to generate responses."
    )


# --------------------------------
# Main Chat Interface
# --------------------------------

st.title("💬 AI Chatbot")

st.caption("Ask me anything. I'm here to help!")

st.divider()


# Check API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("Groq API key is missing. Please configure your .env file.")
    st.stop()


# Initialize Graph
try:
    graph = create_graph(selected_model)

except Exception as e:
    st.error(f"Unable to initialize chatbot: {e}")
    st.stop()


# --------------------------------
# Display Previous Messages
# --------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------
# Chat Input
# --------------------------------

user_prompt = st.chat_input("Type your message here...")


if user_prompt:

    # Convert previous conversation into LangChain messages
    messages = []

    for message in st.session_state.chat_history:

        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        else:
            messages.append(AIMessage(content=message["content"]))

    # Add current user message
    messages.append(HumanMessage(content=user_prompt))

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = graph.invoke({"messages": messages})

                # Get the latest AI response
                response = result["messages"][-1].content

                st.markdown(response)

                # Save conversation
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_prompt
                })

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:

                st.error(f"Something went wrong: {e}")