import streamlit as st
import time
from open_strawberry import manage_conversation, system_prompt, initial_prompt, next_prompts, NUM_TURNS, show_next

st.title("Open Strawberry Conversation")

# Initialize session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "claude-3-haiku-20240307"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False
if "waiting_for_continue" not in st.session_state:
    st.session_state.waiting_for_continue = False
if "generator" not in st.session_state:
    st.session_state.generator = None  # Store the generator in session state
if "prompt" not in st.session_state:
    st.session_state.prompt = None  # Store the prompt in session state
if "output_tokens" not in st.session_state:
    st.session_state.output_tokens = 0
if "input_tokens" not in st.session_state:
    st.session_state.input_tokens = 0
if "cache_creation_input_tokens" not in st.session_state:
    st.session_state.cache_creation_input_tokens = 0
if "cache_read_input_tokens" not in st.session_state:
    st.session_state.cache_read_input_tokens = 0


# Function to display chat messages
def display_chat():
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            assistant_container1 = st.chat_message("assistant")
            with assistant_container1.container():
                st.markdown(message["content"])
        elif message["role"] == "user":
            if not message["initial"] and not show_next:
                continue
            user_container1 = st.chat_message("user")
            with user_container1:
                st.markdown(message["content"])

    # Add a dummy element at the end to ensure scrolling to the latest message
    st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)
    st.markdown("""
        <script>
            var bottom = document.getElementById('bottom');
            if (bottom) {
                bottom.scrollIntoView({behavior: 'smooth'});
            }
        </script>
    """, unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Controls")

# Model selection
st.sidebar.selectbox("Select Model", ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"], key="openai_model")

# Reset conversation button
reset_clicked = st.sidebar.button("Reset Conversation")

if reset_clicked:
    st.session_state.messages = []
    st.session_state.turn_count = 0
    st.sidebar.write(f"Turn count: {st.session_state.turn_count}")
    st.session_state.input_key += 1
    st.session_state.conversation_started = False
    st.session_state.generator = None  # Reset the generator
    reset_clicked = False
    st.session_state.output_tokens = 0
    st.session_state.input_tokens = 0
    st.session_state.cache_creation_input_tokens = 0
    st.session_state.cache_read_input_tokens = 0
    st.rerun()

st.session_state.waiting_for_continue = False

# Display debug information
st.sidebar.write(f"Turn count: {st.session_state.turn_count}")
st.sidebar.write(f"Number of messages: {len(st.session_state.messages)}")
st.sidebar.write(f"Conversation started: {st.session_state.conversation_started}")
st.sidebar.write(f"Output tokens: {st.session_state.output_tokens}")
st.sidebar.write(f"Input tokens: {st.session_state.input_tokens}")
st.sidebar.write(f"Cache creation input tokens: {st.session_state.cache_creation_input_tokens}")
st.sidebar.write(f"Cache read input tokens: {st.session_state.cache_read_input_tokens}")

# Handle user input
if not st.session_state.conversation_started:
    if not st.button("Start Conversation"):
        prompt = st.text_area("What would you like to ask?", value=initial_prompt,
                              key=f"input_{st.session_state.input_key}", height=500)
        st.session_state.prompt = prompt
    else:
        st.session_state.conversation_started = True
        st.session_state.input_key += 1
else:
    assert st.session_state.generator is not None

# Display chat history
chat_container = st.container()
with chat_container:
    display_chat()

# Process conversation
current_assistant_message = ''
assistant_placeholder = None

try:
    while True:
        if st.session_state.waiting_for_continue:
            time.sleep(0.1)  # Short sleep to prevent excessive CPU usage
            continue
        if not st.session_state.conversation_started:
            time.sleep(0.1)
            continue
        elif st.session_state.generator is None:
            st.session_state.generator = manage_conversation(
                model=st.session_state["openai_model"],
                system=system_prompt,
                initial_prompt=st.session_state.prompt,
                next_prompts=next_prompts,
                num_turns=NUM_TURNS,
                yield_prompt=True,
            )
        chunk = next(st.session_state.generator)
        if chunk["role"] == "assistant":
            current_assistant_message += chunk["content"]
            if assistant_placeholder is None:
                assistant_placeholder = st.empty()  # Placeholder for assistant's message

            # Update the assistant container with the progressively streaming message
            with assistant_placeholder.container():
                st.chat_message("assistant").markdown(current_assistant_message)  # Update in the same chat message

        elif chunk["role"] == "user":
            if current_assistant_message:
                st.session_state.messages.append({"role": "assistant", "content": current_assistant_message})
            # Reset assistant message when user provides input
            # Display user message
            if not chunk["initial"] and not show_next:
                pass
            else:
                user_container = st.chat_message("user")
                with user_container:
                    st.markdown(chunk["content"])
            st.session_state.messages.append({"role": "user", "content": chunk["content"], 'initial': chunk["initial"]})

            st.session_state.turn_count += 1
            if current_assistant_message:
                assistant_placeholder = st.empty()  # Reset placeholder
                current_assistant_message = ""

        elif chunk["role"] == "action":
            if chunk["content"] in ["continue?"]:
                # Continue conversation button
                continue_clicked = st.button("Continue Conversation")
                st.session_state.waiting_for_continue = True
            st.session_state.turn_count += 1
            if current_assistant_message:
                st.session_state.messages.append({"role": "assistant", "content": current_assistant_message})
                assistant_placeholder = st.empty()  # Reset placeholder
                current_assistant_message = ""
            elif chunk["content"] == "end":
                break
        elif chunk["role"] == "usage":
            st.session_state.output_tokens += chunk["content"]["output_tokens"]
            st.session_state.input_tokens += chunk["content"]["input_tokens"]
            st.session_state.cache_creation_input_tokens += chunk["content"]["cache_creation_input_tokens"]
            st.session_state.cache_read_input_tokens += chunk["content"]["cache_read_input_tokens"]

        time.sleep(0.005)  # Small delay to prevent excessive updates

except StopIteration:
    pass
