import os

import streamlit as st
import time

try:
    from src.models import get_model_names
    from src.open_strawberry import get_defaults, manage_conversation
except (ModuleNotFoundError, ImportError):
    from models import get_model_names
    from open_strawberry import get_defaults, manage_conversation

(model, system_prompt, initial_prompt, expected_answer,
 next_prompts, num_turns, show_next, final_prompt,
 temperature, max_tokens,
 num_turns_final_mod,
 show_cot,
 verbose) = get_defaults()

st.title("Open Strawberry Conversation")
st.markdown("[Open Strawberry GitHub Repo](https://github.com/pseudotensor/open-strawberry)")

# Initialize session state
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
if "answer" not in st.session_state:
    st.session_state.answer = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = None
if "output_tokens" not in st.session_state:
    st.session_state.output_tokens = 0
if "input_tokens" not in st.session_state:
    st.session_state.input_tokens = 0
if "cache_creation_input_tokens" not in st.session_state:
    st.session_state.cache_creation_input_tokens = 0
if "cache_read_input_tokens" not in st.session_state:
    st.session_state.cache_read_input_tokens = 0
if "verbose" not in st.session_state:
    st.session_state.verbose = verbose
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = max_tokens
if "seed" not in st.session_state:
    st.session_state.seed = 0
if "temperature" not in st.session_state:
    st.session_state.temperature = temperature
if "next_prompts" not in st.session_state:
    st.session_state.next_prompts = next_prompts
if "final_prompt" not in st.session_state:
    st.session_state.final_prompt = final_prompt


# Function to display chat messages
def display_chat():
    display_step = 1
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            if 'final' in message and message['final']:
                display_final(message)
            elif 'turn_title' in message and message['turn_title']:
                display_turn_title(message, display_step=display_step)
                display_step += 1
            else:
                with st.expander("Chain of Thoughts", expanded=st.session_state["show_cot"]):
                    assistant_container1 = st.chat_message("assistant")
                    with assistant_container1.container():
                        st.markdown(message["content"].replace('\n', '  \n'), unsafe_allow_html=True)
        elif message["role"] == "user":
            if not message["initial"] and not st.session_state.show_next:
                continue
            user_container1 = st.chat_message("user")
            with user_container1:
                st.markdown(message["content"].replace('\n', '  \n'), unsafe_allow_html=True)


def display_final(chunk1, can_rerun=False):
    if 'final' in chunk1 and chunk1['final']:
        if st.session_state.answer:
            if st.session_state.answer.strip() in chunk1["content"]:
                st.markdown(f'<h3 class="expander-title">🏆 Final Answer</h3>', unsafe_allow_html=True)
            else:
                st.markdown(f'Expected: **{st.session_state.answer.strip()}**', unsafe_allow_html=True)
                st.markdown(f'<h3 class="expander-title">👎 Final Answer</h3>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h3 class="expander-title">👌 Final Answer</h3>', unsafe_allow_html=True)
        final = chunk1["content"].strip().replace('\n', '  \n')
        if '\n' in final or '<br>' in final:
            st.markdown(f'{final}', unsafe_allow_html=True)
        else:
            st.markdown(f'**{final}**', unsafe_allow_html=True)
        if can_rerun:
            # rerun to get token stats
            st.rerun()


def display_turn_title(chunk1, display_step=None):
    if display_step is None:
        display_step = st.session_state.turn_count
        name = "Completed Step"
    else:
        name = "Step"
    if 'turn_title' in chunk1 and chunk1['turn_title']:
        turn_title = chunk1["content"].strip().replace('\n', '  \n')
        step_time = f' in time {str(int(chunk1["thinking_time"]))}s'
        acum_time = f' in total {str(int(chunk1["total_thinking_time"]))}s'
        st.markdown(f'**{name} {display_step}: {turn_title}{step_time}{acum_time}**', unsafe_allow_html=True)


if st.button("Start Reasoning Engine", disabled=st.session_state.conversation_started):
    st.session_state.conversation_started = True

# Sidebar
st.sidebar.title("Controls")

on_hf_spaces = os.getenv("HF_SPACES", '0') == '1'


def save_env_vars(env_vars):
    assert not on_hf_spaces, "Cannot save env vars in HF Spaces"
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    from dotenv import set_key
    for key, value in env_vars.items():
        set_key(env_path, key, value)


def get_dotenv_values():
    if on_hf_spaces:
        return st.session_state.secrets
    else:
        from dotenv import dotenv_values
        return dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))


if 'secrets' not in st.session_state:
    if on_hf_spaces:
        # allow user to enter
        st.session_state.secrets = dict(OPENAI_API_KEY='',
                                        OPENAI_BASE_URL='https://api.openai.com/v1',
                                        OPENAI_MODEL_NAME='',
                                        # OLLAMA_OPENAI_API_KEY='',
                                        # OLLAMA_OPENAI_BASE_URL='http://localhost:11434/v1/',
                                        # OLLAMA_OPENAI_MODEL_NAME='',
                                        # AZURE_OPENAI_API_KEY='',
                                        # AZURE_OPENAI_API_VERSION='',
                                        # AZURE_OPENAI_ENDPOINT='',
                                        # AZURE_OPENAI_DEPLOYMENT='',
                                        # AZURE_OPENAI_MODEL_NAME='',
                                        GEMINI_API_KEY='',
                                        # MISTRAL_API_KEY='',
                                        GROQ_API_KEY='',
                                        CEREBRAS_OPENAI_API_KEY='',
                                        ANTHROPIC_API_KEY='',
                                        )

    else:
        st.session_state.secrets = {}


def update_model_selection():
    visible_models1 = get_model_names(st.session_state.secrets, on_hf_spaces)
    if visible_models1 and "model_name" in st.session_state:
        if st.session_state.model_name not in visible_models1:
            st.session_state.model_name = visible_models1[0]


# Replace the existing model selection code with this
if 'model_name' not in st.session_state or not st.session_state.model_name:
    update_model_selection()

# Model selection
visible_models = get_model_names(st.session_state.secrets, on_hf_spaces)
st.sidebar.selectbox("Select Model", visible_models, key="model_name",
                     disabled=st.session_state.conversation_started)
st.sidebar.checkbox("Show Next", value=show_next, key="show_next", disabled=st.session_state.conversation_started)
st.sidebar.number_input("Num Turns to Check if Final Answer", value=num_turns_final_mod, key="num_turns_final_mod",
                        disabled=st.session_state.conversation_started)
st.sidebar.number_input("Num Turns per User Click of Continue", value=num_turns, key="num_turns",
                        disabled=st.session_state.conversation_started)
st.sidebar.checkbox("Show Chain of Thoughts Details", value=show_cot, key="show_cot",
                    disabled=st.session_state.conversation_started)

# Reset conversation button
reset_clicked = st.sidebar.button("Reset Conversation")
with st.sidebar.expander("Edit in-memory session secrets" if on_hf_spaces else "Edit .env", expanded=on_hf_spaces):
    dotenv_dict = get_dotenv_values()
    new_env = {}
    for k, v in dotenv_dict.items():
        new_env[k] = st.text_input(k, value=v, key=k, disabled=st.session_state.conversation_started, type="password")
        st.session_state.secrets[k] = new_env[k]
    save_secrets_clicked = st.button("Save dotenv" if not on_hf_spaces else "Save secrets to memory")

    if save_secrets_clicked:
        if on_hf_spaces:
            st.success("secrets temporarily stored to your session memory only")
        else:
            save_env_vars(st.session_state.user_secrets)
            st.success("dotenv saved to .env file")

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
num_messages = len([x for x in st.session_state.messages if x.get('role', '') == 'assistant'])
st.sidebar.write(f"Number of AI messages: {num_messages}")
st.sidebar.write(f"Conversation started: {st.session_state.conversation_started}")
st.sidebar.write(f"Output tokens: {st.session_state.output_tokens}")
st.sidebar.write(f"Input tokens: {st.session_state.input_tokens}")
st.sidebar.write(f"Cache creation input tokens: {st.session_state.cache_creation_input_tokens}")
st.sidebar.write(f"Cache read input tokens: {st.session_state.cache_read_input_tokens}")

# Handle user input
if not st.session_state.conversation_started:
    prompt = st.text_area("What would you like to ask?", value=initial_prompt,
                          key=f"input_{st.session_state.input_key}", height=500)
    st.session_state.prompt = prompt
    answer = st.text_area("Expected answer (Empty if do not know)", value=expected_answer,
                          key=f"answer_{st.session_state.input_key}", height=100)
    st.session_state.answer = answer
    system_prompt = st.text_area("System Prompt", value=system_prompt,
                                 key=f"system_prompt_{st.session_state.input_key}", height=200)
    st.session_state.system_prompt = system_prompt
else:
    st.session_state.conversation_started = True
    st.session_state.input_key += 1

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
                model=st.session_state["model_name"],
                system=st.session_state.system_prompt,
                initial_prompt=st.session_state.prompt,
                next_prompts=st.session_state.next_prompts,
                final_prompt=st.session_state.final_prompt,
                num_turns_final_mod=st.session_state.num_turns_final_mod,
                num_turns=st.session_state.num_turns,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                seed=st.session_state.seed,
                secrets=st.session_state.secrets,
                verbose=st.session_state.verbose,
            )
        chunk = next(st.session_state.generator)
        if chunk["role"] == "assistant":
            if not chunk.get('final', False) and not chunk.get('turn_title', False):
                current_assistant_message += chunk["content"]
            if assistant_placeholder is None:
                assistant_placeholder = st.empty()  # Placeholder for assistant's message

            # Update the assistant container with the progressively streaming message
            with assistant_placeholder.container():
                # Update in the same chat message
                with st.expander("Chain of Thoughts", expanded=st.session_state["show_cot"]):
                    st.chat_message("assistant").markdown(current_assistant_message, unsafe_allow_html=True)
                if 'turn_title' in chunk and chunk['turn_title']:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": chunk['content'], 'turn_title': True,
                         'thinking_time': chunk['thinking_time'],
                         'total_thinking_time': chunk['total_thinking_time']})
                    display_turn_title(chunk)
                if 'final' in chunk and chunk['final']:
                    # user role would normally do this, but on final step needs to be here
                    st.session_state.messages.append(
                        {"role": "assistant", "content": current_assistant_message, 'final': False})
                    # last message, so won't reach user turn, so need to store final assistant message from parsing
                    st.session_state.messages.append(
                        {"role": "assistant", "content": chunk['content'], 'final': True})
                    display_final(chunk, can_rerun=True)

        elif chunk["role"] == "user":
            if current_assistant_message:
                st.session_state.messages.append(
                    {"role": "assistant", "content": current_assistant_message, 'final': chunk.get('final', False)})
            # Reset assistant message when user provides input
            # Display user message
            if not chunk["initial"] and not st.session_state.show_next:
                pass
            else:
                user_container = st.chat_message("user")
                with user_container:
                    st.markdown(chunk["content"].replace('\n', '  \n'), unsafe_allow_html=True)
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
            st.session_state.output_tokens += chunk["content"]["output_tokens"] if "output_tokens" in chunk[
                "content"] else 0
            st.session_state.input_tokens += chunk["content"]["input_tokens"] if "input_tokens" in chunk[
                "content"] else 0
            st.session_state.cache_creation_input_tokens += chunk["content"][
                "cache_creation_input_tokens"] if "cache_creation_input_tokens" in chunk["content"] else 0
            st.session_state.cache_read_input_tokens += chunk["content"][
                "cache_read_input_tokens"] if "cache_read_input_tokens" in chunk["content"] else 0

        time.sleep(0.001)  # Small delay to prevent excessive updates

except StopIteration:
    pass
