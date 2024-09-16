import streamlit as st

from open_strawberry import (
    manage_conversation,
    system_prompt,
    initial_prompt
)

st.title("Open Strawberry Conversation")

# Display the imported system prompt
st.text_area("System Prompt", value=system_prompt, height=300, disabled=True)

# Display the imported initial prompt, but allow it to be edited
user_prompt = st.text_input("Initial Prompt", value=initial_prompt)

# Model selection
model = st.selectbox("Select Model", ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"])

# Initialize session state
if 'conversation_generator' not in st.session_state:
    st.session_state.conversation_generator = None

# Start conversation button
if st.button("Start/Continue Conversation"):
    if st.session_state.conversation_generator is None:
        st.session_state.conversation_generator = manage_conversation(model, system_prompt, user_prompt)

    conversation_active = True
    while conversation_active:
        try:
            message = next(st.session_state.conversation_generator)
            if message["role"] == "user":
                st.write(f"Human: {message['content']}")
            elif message["role"] == "assistant":
                if message.get("streaming", False):
                    message_placeholder = st.empty()
                    full_response = ""
                    full_response += message["content"]
                    message_placeholder.markdown(full_response + "▌")
                else:
                    st.write(f"Assistant: {message['content']}")
            elif message["role"] == "system":
                if message["content"] == "pause":
                    conversation_active = False
                elif message["content"] == "continue?":
                    user_input = st.button("Continue?")
                    st.session_state.conversation_generator.send(user_input)
                elif message["content"] == "end":
                    st.write("Conversation ended.")
                    st.session_state.conversation_generator = None
                    conversation_active = False
        except StopIteration:
            st.write("Conversation ended.")
            st.session_state.conversation_generator = None
            conversation_active = False

# Reset conversation button
if st.button("Reset Conversation"):
    st.session_state.conversation_generator = None
    st.experimental_rerun()
