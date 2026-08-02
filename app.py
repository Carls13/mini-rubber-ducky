import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Define the model
MODEL = "gpt-4o-mini"

# Set the Page configuration
st.set_page_config(
    page_title="Mini Rubber Ducky",
    page_icon=":duck:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Mini Rubber Ducky :duck:")
st.write(
    "Welcome to Mini Rubber Ducky! This is a simple Streamlit app that uses OpenAI's API to generate text based on your input. Enter a prompt below and click 'Generate' to see the magic happen!"
)

# Initialize session state for user input and generated text
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = None
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None


# Function to load the vector store ID from environment variables
def load_vector_store():
    """Load the vector store ID from environment variables and store it in session state."""
    try:
        vector_store_id = os.getenv("VECTOR_STORE_ID")

        if not vector_store_id:
            try:
                vector_store_id = st.secrets.get("VECTOR_STORE_ID")
            except Exception:  # noqa: BLE001, S110
                pass

        if not vector_store_id:
            st.error(
                "Vector store ID not found. Please set VECTOR_STORE_ID in your .env file or Streamlit secrets."
            )
            return None

        return vector_store_id

    except Exception as e:  # noqa: BLE001
        st.error(f"Error loading vector store ID: {e}")
        return None


# Load the vector store ID and store it in session state
st.session_state.vector_store_id = load_vector_store()


# Define the initial message for the bot
INITIAL_MESSAGE = """
Hi! I'm your Mini Rubber Ducky assistant. 
How can I help you today?
I can help you in many modes
"""


# Instructions for the bot
INSTRUCTIONS = """ 
You are Mini Rubber Ducky, the course assistant for a multimodal RAG assistant course.
Be concise, direct, and practical. Use active voice. No fluff.

Primary objective
- Answer questions about the course content and code using the attached Vector Store (transcripts, notebooks, scripts).
- Prefer retrieved facts over memory. If the files don't cover it, say so.
- Focus on multimodal RAG concepts, implementations, and best practices.

Retrieval & citations
- Always use File Search first.
- Ground every substantive answer in retrieved snippets.
- If nothing relevant is found, say: "I don't see this in the course files." Then suggest the most relevant module(s) the learner should review.
- Never include source citations or reference labels in the final answer text.

Answer style
- Keep outputs scannable: short paragraphs, bullet steps, compact runnable code samples when needed.
- When explaining "how to build X", outline the pipeline stages (ingest → retrieve → generate → evaluate) before diving into code.
- Close each reply with a friendly follow-up question the learner might ask next.
- Stay approachable, encouraging, and human.

Boundaries
- Don't invent references, credentials, metrics, or file names.
- If the topic is outside multimodal RAG/this curriculum, acknowledge the gap and offer a high-level pointer or ask for clarification.

Context: Course focus
- Multimodal RAG: handling text, images, audio, and video in retrieval-augmented generation systems
- Vector stores and embeddings for multimodal data
- Integration patterns and architectures
- Practical implementations and code examples

If the learner references a lecture/section by name/number, search for files with that stem and tailor the answer.
Never invent lecture numbers or titles—they change over time.
If the answer isn't in the corpus, say so clearly.
"""


# Function to ask the bot a question
def ask_bot(user_prompt: str):
    """Send questions to OpenAI and get responses"""
    common_kwargs = {
        "model": MODEL,
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [st.session_state.vector_store_id],
                "max_num_results": 20,
            }
        ],
        "text": {"verbosity": "medium"},
        "instructions": INSTRUCTIONS,
    }

    if st.session_state.previous_response_id:
        resp = client.responses.create(
            previous_response_id=st.session_state.previous_response_id,
            input=[{"role": "user", "content": user_prompt}],
            **common_kwargs,
        )
    else:
        resp = client.responses.create(
            input=[
                {"role": "assistant", "content": INITIAL_MESSAGE.strip()},
                {"role": "user", "content": user_prompt},
            ],
            **common_kwargs,
        )

    # Update the previous response ID in session state
    st.session_state.previous_response_id = resp.id

    return resp.output_text.strip() if resp.output_text else "No response generated."


# Function to reset the conversation
def reset_conversation():
    """Reset the conversation by clearing the messages in session state."""
    st.session_state.messages = [
        {"role": "assistant", "content": INITIAL_MESSAGE.strip()}
    ]
    st.success("Conversation has been reset.")
    st.rerun()


def main():
    # Sidebar with reset button
    with st.sidebar:
        st.header("Settings")
        if st.button("Reset Conversation"):
            reset_conversation()

    # Load the vector store ID and store it in session state
    if not st.session_state.vector_store_id:
        st.session_state.vector_store_id = load_vector_store()

    # Initialize the messages
    if not st.session_state.messages:
        st.session_state.messages = [
            {"role": "assistant", "content": INITIAL_MESSAGE.strip()}
        ]

    # Display the entire conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    prompt = st.chat_input("Ask me anything about multimodal RAG")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the user input and generate a response only after the user
        # submits a prompt. On the initial page load, `prompt` is None.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_bot(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
