import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda

# Initialize AI Model
chat_model = ChatGoogleGenerativeAI(
    google_api_key="AIzaSyC_gR124QEGp4tWdjWIPrclyW9vucxo7GQ",
    model="gemini-1.5-pro",
    temperature=1
)

# Define Chat Template
chat_template = ChatPromptTemplate(
    messages=[
        ("system", "👨‍🏫 You are an AI Data Science Tutor. "
                   "You must answer ONLY Data Science-related questions. "
                   "If the user asks non-data science questions, politely refuse and redirect them to relevant topics. "
                   "Provide detailed technical explanations in simple terms. "
                   "When relevant, include code snippets and AI-generated images to enhance understanding. "
                   "Ensure that all code is clean, efficient, and uses best practices. "
                   "For coding questions, always include proper syntax, explanations, and example outputs. "
                   "For visualization-related topics, generate appropriate images using AI."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

output_parser = StrOutputParser()
memory = ConversationBufferMemory(return_messages=True)

def get_history_and_input(user_input):
    return {
        "chat_history": memory.chat_memory.messages,
        "human_input": user_input
    }

def get_history(_=None):
    return {"chat_history": memory.chat_memory.messages}

chain = (
    RunnableLambda(lambda x: get_history_and_input(x["human_input"]))
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
st.set_page_config(page_title="🤖 Data Science Chatbot", layout="wide")

st.title("Suman Data Science AI Chatbot")
st.subheader("Ask your doubts here")


# Chat History UI
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    role, content = message
    if role == "user":
        st.markdown(f"👤 **You:** {content}")
    else:
        st.markdown(f"🤖 **AI:** {content}")

# User Input
user_input = st.text_input("💬 Write ✍🏻 your Message:", key="user_input")

if st.button("Enter") and user_input:
    # Display user message
    st.session_state["messages"].append(("user", user_input))

    # Get AI response
    query = {"human_input": user_input}
    response = chain.invoke(query)

    # Display AI response
    st.session_state["messages"].append(("ai", response))
    st.markdown(f"🤖 **AI:** {response}")

    # Save to memory
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)