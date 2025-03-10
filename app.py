import streamlit as st
import google.generativeai as ai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import BaseLLM

# Title of Streamlit app
st.title("Suman Data Science Tutor Chatbot")

# API key setup (Make sure to use your actual API key here)
api_key = "AIzaSyC_gR124QEGp4tWdjWIPrclyW9vucxo7GQ"

# Validating API key
if api_key == "your_actual_api_key_here" or not api_key:
    st.error("API key is not available or invalid.")
    st.stop()

# Configure Google Gemini API with the API key
ai.configure(api_key=api_key)

# Create a custom LLM wrapper for Gemini 1.5 Pro
class GeminiLLM(BaseLLM):
    def __init__(self, model_name="models/gemini-1.5-pro"):
        self.model_name = model_name
    
    def _call(self, prompt: str) -> str:
        # Generate the response from Google's Gemini API
        try:
            response = ai.GenerativeModel(model_name=self.model_name).generate_content(prompt)
            return response.text
        except Exception as e:
            raise ValueError(f"Error generating response from Gemini model: {str(e)}")
    
    def _llm_type(self) -> str:
        return "google_gemini"

# Initialize the custom Gemini model
gemini_llm = GeminiLLM(model_name="models/gemini-1.5-pro")

# Define the system prompt for the assistant
sys_prompt = """
You are a helpful AI tutor for data science.
Students will ask you doubts related to various topics in Data Science.
You are expected to reply in as much detail as possible.
Make sure to take examples while explaining the concepts.
In case a student asks any question outside the data science scope,
politely decline and tell them to ask questions within the data science domain only.
"""

# Initialize memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the conversation chain
conversation = ConversationChain(
    llm=gemini_llm, 
    memory=memory,
    verbose=True,
    prompt=PromptTemplate(input_variables=["question"], template=sys_prompt)
)

# Initialize session state for storing conversation history if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Personalized Greeting
if len(st.session_state.messages) == 0:
    st.markdown(""" 
    ## Hello, welcome to the Data Science Tutor Chatbot! 👋

    Hi, this is **Suman** talking to you, and I am your AI tutor. Feel free to ask me any questions related to Data Science! 😊

    I'm here to help you with your doubts. Let's get started!
    """)

# Function to display conversation history
def display_chat():
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['text']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Suman (Tutor):** {message['text']}")

# Display previous conversation
display_chat()

# User input field
user_input = st.text_input("Enter your question:", placeholder="Ask your question here...")

if user_input.strip():  # If the user has entered some text
    # Append user input to the chat history in session state
    st.session_state.messages.append({"role": "user", "text": user_input})

    # Generate response using Langchain
    try:
        # Send the user query to the conversation chain and get the assistant's response
        assistant_response = conversation.predict(input=user_input)

        # Append assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "text": assistant_response})

        # Display the updated chat
        display_chat()

    except Exception as e:
        st.error(f"Error generating content: {str(e)}")

