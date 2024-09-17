import os
import random
from typing import List, Dict, Generator

# https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
import anthropic

clawd_key = os.getenv('ANTHROPIC_API_KEY')
clawd_client = anthropic.Anthropic(api_key=clawd_key) if clawd_key else None

random.seed(1234)

show_next = False  # CHOOSE: True to show all messages, False to show only assistant messages
verbose = False  # CHOOSE: True to show usage information, False to hide it


def get_anthropic(model: str,
                  prompt: str,
                  temperature: float = 0,
                  max_tokens: int = 1024,
                  system: str = '',
                  chat_history: List[Dict] = None) -> \
        Generator[dict, None, None]:
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

    output_tokens = 0
    input_tokens = 0
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0
    for chunk in response:
        if chunk.type == "content_block_start":
            # This is where we might find usage info in the future
            pass
        elif chunk.type == "content_block_delta":
            yield dict(text=chunk.delta.text)
        elif chunk.type == "message_delta":
            output_tokens = dict(chunk.usage).get('output_tokens', 0)
        elif chunk.type == "message_start":
            usage = chunk.message.usage
            input_tokens = dict(usage).get('input_tokens', 0)
            cache_creation_input_tokens = dict(usage).get('cache_creation_input_tokens', 0)
            cache_read_input_tokens = dict(usage).get('cache_read_input_tokens', 0)
        else:
            if verbose:
                print("Unknown chunk type:", chunk.type)
                print("Chunk:", chunk)

    if verbose:
        # After streaming is complete, print the usage information
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens: {input_tokens}")
        print(f"Cache creation input tokens: {cache_creation_input_tokens}")
        print(f"Cache read input tokens: {cache_read_input_tokens}")
    yield dict(output_tokens=output_tokens, input_tokens=input_tokens,
               cache_creation_input_tokens=cache_creation_input_tokens,
               cache_read_input_tokens=cache_read_input_tokens)


NUM_TURNS = int(os.getenv('NUM_TURNS', '10'))  # Number of turns before pausing for continuation


def manage_conversation(model: str,
                        system: str,
                        initial_prompt: str,
                        next_prompts: List[str],
                        num_turns: int = NUM_TURNS,
                        cli_mode: bool = False,
                        yield_prompt=True) -> Generator[Dict, None, list]:
    chat_history = []

    # Initial prompt
    if yield_prompt:
        yield {"role": "user", "content": initial_prompt, "chat_history": chat_history, "initial": True}
    response_text = ''
    for chunk in get_anthropic(model, initial_prompt, system=system):
        if 'text' in chunk and chunk['text']:
            response_text += chunk['text']
            yield {"role": "assistant", "content": chunk['text'], "streaming": True, "chat_history": chat_history}
        else:
            yield {"role": "usage", "content": chunk}

    chat_history.append(
        {"role": "user", "content": [{"type": "text", "text": initial_prompt, "cache_control": {"type": "ephemeral"}}]})
    chat_history.append({"role": "assistant", "content": response_text})

    turn_count = 1

    while True:
        next_prompt = random.choice(next_prompts)
        yield {"role": "user", "content": next_prompt, "chat_history": chat_history, "initial": False}

        response_text = ''
        for chunk in get_anthropic(model, next_prompt, system=system, chat_history=chat_history):
            if 'text' in chunk and chunk['text']:
                response_text += chunk['text']
                yield {"role": "assistant", "content": chunk['text'], "streaming": True, "chat_history": chat_history}
            else:
                yield {"role": "usage", "content": chunk}

        chat_history.append(
            {"role": "user",
             "content": [{"type": "text", "text": next_prompt, "cache_control": {"type": "ephemeral"}}]})
        chat_history.append({"role": "assistant", "content": response_text})

        turn_count += 1

        if turn_count % num_turns == 0:
            if cli_mode:
                user_continue = input("Continue? (y/n): ").lower() == 'y'
                if not user_continue:
                    break
            else:
                yield {"role": "action", "content": "continue?", "chat_history": chat_history}


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

next_prompts = ["next",
                "next",
                "next",
                "Are you sure?",
                "How would you verify your answer?",
                "Any mistakes?",
                "Go back 3 steps and try again.",
                "Take a deep breath and work on this problem step-by-step.",
                "Break this down.",
                "Please ensure you think from first principles.",
                """List a much more general abstract versions of the original question, then describe the situation using your imagination ensuring not to over-constrain the problem, then explore in a list all the possible different constraints or lack of constraints (be sure to consider from a human viewpoint) relevant for the circumstance, then explore in a list the many extreme possibilities for issues. Let's work this out in a well-structured step-by-step thoughtful way to be sure we have the right answer. Make a final best guess using common sense.""",
                """1) Restate the original question in elaborate form.
2) Give an abstract version of the original question.
3) Provide a detailed highly-accurate and well-structured response to the user's original question.
4) Give a detailed highly-accurate and well-structured justification for the response.
5) Evaluate your response with a score of 0 through 10.  10 means the justification perfectly explains the response to the original question and the response is perfectly accurate, 5 means the response and justification might contain some errors, 0 means the response is not accurate or is not well-justified.
"""
                ]


def go():
    # model = "claude-3-5-sonnet-20240620"
    model = "claude-3-haiku-20240307"
    generator = manage_conversation(model=model, system=system_prompt,
                                    initial_prompt=initial_prompt,
                                    next_prompts=next_prompts,
                                    num_turns=10,
                                    cli_mode=True)
    response = ''
    conversation_history = []

    try:
        while True:
            chunk = next(generator)
            if 'role' in chunk and chunk['role'] == 'assistant':
                response += chunk['content']
                conversation_history = chunk['chat_history']
                print(chunk['content'], end='')
            elif 'role' in chunk and chunk['role'] == 'user':
                if not chunk['initial'] and not show_next:
                    continue
                print('\n', end='')  # finish assistant
                print('\nUser: ', chunk['content'], end='\n\n')
                print('\nAssistant:\n\n ')
    except StopIteration as e:
        pass

    print("Conversation history:", conversation_history)


if __name__ == '__main__':
    go()
