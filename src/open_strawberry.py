import os
from typing import List, Dict, Generator

# https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
import anthropic

clawd_key = os.getenv('ANTHROPIC_API_KEY')
clawd_client = anthropic.Anthropic(api_key=clawd_key) if clawd_key else None


def get_anthropic(model: str,
                  prompt: str,
                  temperature: float = 0,
                  max_tokens: int = 1024,
                  system: str = '',
                  chat_history: List[Dict] = None) -> \
        Generator[str, None, None]:
    if chat_history is None:
        chat_history = []

    messages = []

    # Add conversation history, removing cache_control from all but the last two user messages
    for i, message in enumerate(chat_history):
        if message["role"] == "user":
            if i >= len(chat_history) - 3:  # Last two user messages
                messages.append(message)
            else:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": message["content"][0]["text"]}]
                })
        else:
            messages.append(message)

    # Add the new user message
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    })

    response = clawd_client.beta.prompt_caching.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
        stream=True
    )

    for chunk in response:
        if chunk.type == "content_block_delta":
            yield chunk.delta.text


def manage_conversation(model: str, system: str, initial_prompt: str, num_turns: int = 10, cli_mode: bool = False) -> \
Generator[
    Dict, None, list]:
    chat_history = []

    # Initial prompt
    yield {"role": "user", "content": initial_prompt}
    for chunk in get_anthropic(model, initial_prompt, system=system):
        yield {"role": "assistant", "content": chunk, "streaming": True}

    response_text = "".join(chunk for chunk in get_anthropic(model, initial_prompt, system=system))
    chat_history.append(
        {"role": "user", "content": [{"type": "text", "text": initial_prompt, "cache_control": {"type": "ephemeral"}}]})
    chat_history.append({"role": "assistant", "content": response_text})

    turn_count = 1

    while True:
        yield {"role": "user", "content": "next"}
        for chunk in get_anthropic(model, "next", system=system, chat_history=chat_history):
            yield {"role": "assistant", "content": chunk, "streaming": True}

        response_text = "".join(
            chunk for chunk in get_anthropic(model, "next", system=system, chat_history=chat_history))
        chat_history.append(
            {"role": "user", "content": [{"type": "text", "text": "next", "cache_control": {"type": "ephemeral"}}]})
        chat_history.append({"role": "assistant", "content": response_text})

        turn_count += 1

        if turn_count % num_turns == 0:
            yield {"role": "system", "content": "pause"}
            if cli_mode:
                user_continue = input("Continue? (y/n): ").lower() == 'y'
            else:
                user_continue = yield {"role": "system", "content": "continue?"}
            if not user_continue:
                break

    yield {"role": "system", "content": "end"}
    return chat_history


system_prompt = """Let us play a game of "take only the most minuscule step toward the solution."
<thinking_game>
* The assistant's text output must be only the very next possible step.
* Use your text output as a scratch pad in addition to a literal output of some next step.
* Everytime you make a major shift in thinking, output your high-level current thinking in <thinking> </thinking> XML tags.
* You should present your response in a way that iterates on that scratch pad space with surrounding textual context.
* You win the game is you are able to take the smallest text steps possible while still (on average) heading towards the solution.
* Backtracking is allowed, and generating python code is allowed (but will not be executed, but can be used to think), just on average over many text output turns you must head towards the answer.
* You must think using first principles, and ensure you identify inconsistencies, errors, etc.
</thinking_game>
Are you ready to win the game?"""

# initial_prompt = "Let's solve a complex math problem: Find the roots of the equation x^3 - 6x^2 + 11x - 6 = 0"

initial_prompt = """Can you crack the code?
9 2 8 5 (One number is correct but in the wrong position)
1 9 3 7 (Two numbers are correct but in the wrong positions)
5 2 0 1 (one number is correct and in the right position)
6 5 0 7 (nothing is correct)
8 5 2 4 (two numbers are correct but in the wrong positions)"""


def go():
    # model = "claude-3-5-sonnet-20240620"
    model = "claude-3-haiku-20240307"
    generator = manage_conversation(model=model, system=system_prompt, initial_prompt=initial_prompt, num_turns=10,
                                    cli_mode=True)
    response = ''
    conversation_history = []

    try:
        while True:
            chunk = next(generator)
            if 'role' in chunk and chunk['role'] == 'assistant':
                response += chunk['content']
                print(chunk['content'], end='')
            elif 'role' in chunk and chunk['role'] == 'user':
                print('\n', end='')  # finish assistant
                print('\nUser: ', chunk['content'], end='\n\n')
                print('\nAssistant:\n\n ')
    except StopIteration as e:
        # Get the return value
        if isinstance(e.value, dict):
            conversation_history = e.value

    print("Conversation history:", conversation_history)


if __name__ == '__main__':
    go()
