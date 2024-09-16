import os
from typing import List, Dict

# https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
import anthropic

clawd_key = os.getenv('ANTHROPIC_API_KEY')
clawd_client = anthropic.Anthropic(api_key=clawd_key) if clawd_key else None


def get_anthropic(model: str, prompt: str, temperature: float = 0, system: str = ''):
    client = anthropic.Anthropic(api_key=clawd_key)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]

    response = client.beta.prompt_caching.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=system,
        messages=messages
    )

    return response.content[0].text, response


def manage_conversation(model: str, system: str, initial_prompt: str, num_turns: int = 10):
    conversation_history = []

    # Start with the initial prompt
    response_text, _ = get_anthropic(model, initial_prompt, system=system)
    print("Assistant:", response_text)

    conversation_history.append({"role": "user", "content": initial_prompt})
    conversation_history.append({"role": "assistant", "content": response_text})

    turn_count = 1

    while True:
        # Construct the prompt based on the entire conversation history
        full_prompt = initial_prompt + "\n\n"
        for msg in conversation_history[1:]:  # Skip the initial prompt
            if msg["role"] == "user":
                full_prompt += f"Human: {msg['content']}\n"
            else:
                full_prompt += f"Assistant: {msg['content']}\n"
        full_prompt += "Human: next\n\nAssistant:"

        response_text, _ = get_anthropic(model, full_prompt, system=system)
        print("Assistant:", response_text)

        conversation_history.append({"role": "user", "content": "next"})
        conversation_history.append({"role": "assistant", "content": response_text})

        turn_count += 1

        if turn_count % num_turns == 0:
            user_input = input("Continue? (yes/no): ")
            if user_input.lower() != 'yes':
                break

    return conversation_history


def go():
    system_prompt = """Let us play a game of "take only the most minuscule step toward the solution."
    <thinking_game>
    * The assistant's text output must be only the very next possible step.
    * Use your text output as a scratch pad in addition to a literal output of some next step.
    * Everytime you make a major shift in thinking, output your high-level current thinking in <thinking> </thinking> XML tags.
    * You should present your response in a way that iterates on that scratch pad space with surrounding textual context.
    * You win the game is you are able to take the smallest text steps possible while still (on average) heading towards the solution.
    * Backtracking is allowed, and generating python code is allowed (but will not be executed, but can be used to think), just on average over many text output turns you must head towards the answer.
    * You should think like a human, and ensure you identify inconsistencies, errors, etc.
    </thinking_game>
    Are you ready to win the game?"""

    initial_prompt = "Let's solve a complex math problem: Find the roots of the equation x^3 - 6x^2 + 11x - 6 = 0"
    # model = "claude-3-5-sonnet-20240620"
    model = "claude-3-haiku-20240307"
    conversation_history = manage_conversation(model, system_prompt, initial_prompt)

    print("Conversation history:", conversation_history)


if __name__ == '__main__':
    go()
