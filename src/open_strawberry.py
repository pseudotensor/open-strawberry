import os
import random
import time
from typing import List, Dict, Generator

from src.models import get_model_api
from src.utils import get_turn_title, get_final_answer


def get_last_assistant_responses(chat_history, n=3):
    assistant_messages = [msg['content'] for msg in chat_history if msg['role'] == 'assistant']
    return assistant_messages[-n:]


def manage_conversation(model: str,
                        system: str,
                        initial_prompt: str,
                        next_prompts: List[str],
                        final_prompt: str = "",
                        num_turns: int = 10,
                        num_turns_final_mod: int = 9,
                        cli_mode: bool = False,
                        temperature: float = 0.3,
                        max_tokens: int = 1024,
                        seed: int = 1234,
                        verbose=False,
                        verification_interval: int = 5
                        ) -> Generator[Dict, None, list]:
    if seed == 0:
        seed = random.randint(0, 1000000)
    random.seed(seed)
    get_model_func = get_model_api(model)
    chat_history = []

    turn_count = 0
    total_thinking_time = 0
    while True:
        trying_final = False
        if turn_count % verification_interval == 0 and turn_count > 0:
            # Perform verification step
            last_responses = get_last_assistant_responses(chat_history, n=3)
            verification_prompt = "Please review your previous reasoning steps and check for any mistakes or inconsistencies. If you find any errors, please correct them and explain your corrections. Here are your previous reasoning steps:\n\n" + "\n\n".join(last_responses)
            prompt = verification_prompt
            yield {"role": "user", "content": prompt, "chat_history": chat_history, "verification": True, "initial": False}
        else:
            if turn_count == 0:
                prompt = initial_prompt
            elif turn_count % num_turns_final_mod == 0 and turn_count > 0:
                trying_final = True
                prompt = final_prompt
            else:
                prompt = random.choice(next_prompts)
            yield {"role": "user", "content": prompt, "chat_history": chat_history, "initial": turn_count == 0}

        thinking_time = time.time()
        response_text = ''
        for chunk in get_model_func(model, prompt, system=system, chat_history=chat_history,
                                    temperature=temperature, max_tokens=max_tokens, verbose=verbose):
            if 'text' in chunk and chunk['text']:
                response_text += chunk['text']
                yield {"role": "assistant", "content": chunk['text'], "streaming": True, "chat_history": chat_history,
                       "final": False, "turn_title": False}
            else:
                yield {"role": "usage", "content": chunk}
        thinking_time = time.time() - thinking_time
        total_thinking_time += thinking_time

        turn_title = get_turn_title(response_text)
        yield {"role": "assistant", "content": turn_title, "turn_title": True, 'thinking_time': thinking_time,
               'total_thinking_time': total_thinking_time}

        chat_history.append(
            {"role": "user",
             "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]})
        chat_history.append({"role": "assistant", "content": response_text})

        # Adjusted to only check final answer when trying_final is True
        always_check_final = False
        if trying_final:
            final_value = get_final_answer(response_text, cli_mode=cli_mode)
            if final_value:
                chat_history.append({"role": "assistant", "content": final_value})
                yield {"role": "assistant", "content": final_value, "streaming": True, "chat_history": chat_history,
                       "final": True}
                break

        turn_count += 1

        if turn_count % num_turns == 0:
            if cli_mode:
                user_continue = input("\nContinue? (y/n): ").lower() == 'y'
                if not user_continue:
                    break
            else:
                yield {"role": "action", "content": "continue?", "chat_history": chat_history}
        time.sleep(0.001)


def get_defaults():
    system_prompt = """Let us play a game of "take only the most minuscule step toward the solution."
<thinking_game>
* The assistant's text output must be only the very next possible step.
* Use your text output as a scratch pad in addition to a literal output of some next step.
* Every time you make a major shift in thinking, output your high-level current thinking in <thinking> </thinking> XML tags.
* You should present your response in a way that iterates on that scratch pad space with surrounding textual context.
* You win the game if you are able to take the smallest text steps possible while still (on average) heading towards the solution.
* Backtracking is allowed, and generating python code is allowed (but will not be executed, but can be used to think), just on average over many text output turns you must head towards the answer.
* You must think using first principles, and ensure you identify inconsistencies, errors, etc.
* Periodically, you should review your previous reasoning steps and check for errors or inconsistencies. If you find any, correct them.
* You MUST always end with a very brief natural language title (it should just describe the analysis, do not give step numbers) of what you did inside <turn_title> </turn_title> XML tags. Only a single title is allowed.
* Do not provide the final answer unless the user specifically requests it using the final prompt.
</thinking_game>
Are you ready to win the game?"""

    initial_prompt = """Can you crack the code?
9 2 8 5 (One number is correct but in the wrong position)
1 9 3 7 (Two numbers are correct but in the wrong positions)
5 2 0 1 (one number is correct and in the right position)
6 5 0 7 (nothing is correct)
8 5 2 4 (two numbers are correct but in the wrong positions)"""

    expected_answer = "3841"

    next_prompts = ["continue effort to answering user's original query",
                    "continue effort to answering user's original query",
                    "continue effort to answering user's original query",
                    "Are you sure?",
                    "How would you verify your answer?",
                    "Any mistakes?",
                    "Go back 3 steps and try again.",
                    "Take a deep breath and work on this problem step-by-step.",
                    "Break this down.",
                    "Please ensure you think from first principles.",
                    """Follow these steps:
1) List a much more general abstract versions of the original question, then describe the situation using your imagination ensuring not to over-constrain the problem.
2) Explore in a list all the possible different constraints or lack of constraints (be sure to consider from a human viewpoint) relevant for the circumstance, then explore in a list the many extreme possibilities for issues.
3) Let's work this out in a well-structured step-by-step thoughtful way to be sure we have the right answer.
4) Make a final best guess using common sense.
""",
                    """Follow these steps:
1) Restate the original question in elaborate form.
2) Give an abstract version of the original question.
3) Provide a detailed highly-accurate and well-structured response to the user's original question.
4) Give a detailed highly-accurate and well-structured justification for the response.
5) Evaluate your response with a score of 0 through 10.  10 means the justification perfectly explains the response to the original question and the response is perfectly accurate, 5 means the response and justification might contain some errors, 0 means the response is not accurate or is not well-justified.
"""
                    ]

    final_prompt = """Verification check list:
1) Do you have very high confidence in a final answer?
2) Have you fully verified your answer with all the time and resources you have?
3) If you have very high confidence AND you have fully verified your answer with all resources possible, then put the final answer in <final_answer> </final_answer> XML tags, otherwise please continue to vigorously work on the user's original query.
"""

    num_turns = int(os.getenv('NUM_TURNS', '10'))  # Number of turns before pausing for continuation
    num_turns_final_mod = num_turns - 1  # Not required, just an OK value. Could be randomized.

    show_next = False
    show_cot = False
    verbose = False

    # model = "claude-3-5-sonnet-20240620"
    model = "anthropic:claude-3-haiku-20240307"

    temperature = 0.3
    max_tokens = 1024

    return (model, system_prompt,
            initial_prompt,
            expected_answer,
            next_prompts,
            num_turns, show_next, final_prompt,
            temperature, max_tokens,
            num_turns_final_mod,
            show_cot,
            verbose)


if __name__ == '__main__':
    from src.cli import go_cli

    go_cli()
