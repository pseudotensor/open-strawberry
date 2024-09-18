import argparse
import os
import random
import re
import time
from typing import List, Dict, Generator

from src.models import get_model_api


def parse_arguments(model, system_prompt, next_prompts, num_turns, show_next, final_prompt,
                    num_turns_final_mod, show_cot, verbose):
    parser = argparse.ArgumentParser(description="Open Strawberry Conversation Manager")
    parser.add_argument("--show_next", action="store_true", default=show_next, help="Show all messages")
    parser.add_argument("--verbose", action="store_true", default=verbose, help="Show usage information")
    parser.add_argument("--system_prompt", type=str, default=system_prompt, help="Custom system prompt")
    parser.add_argument("--num_turns_final_mod", type=int, default=num_turns_final_mod,
                        help="Number of turns before final prompt")
    parser.add_argument("--num_turns", type=int, default=num_turns,
                        help="Number of turns before pausing for continuation")
    parser.add_argument("--model", type=str, default=model, help="Model to use for conversation")
    parser.add_argument("--initial_prompt", type=str, default='', help="Initial prompt.  If empty, then ask user.")
    parser.add_argument("--expected_answer", type=str, default='', help="Expected answer.  If empty, then ignore.")
    parser.add_argument("--next_prompts", type=str, nargs="+", default=next_prompts, help="Next prompts")
    parser.add_argument("--final_prompt", type=str, default=final_prompt, help="Final prompt")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for the model")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens for the model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, 0 means random seed")
    parser.add_argument("--show_cot", type=bool, default=show_cot, help="Whether to show detailed Chain of Thoughts")

    return parser.parse_args()


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

        # FIXME: Always check for now, goes too far otherwise sometimes, but that may be good on harder problems.
        if trying_final or True:
            tag = 'final_answer'
            pattern = fr'<{tag}>(.*?)</{tag}>'
            values = re.findall(pattern, response_text, re.DOTALL)
            values0 = values.copy()
            values = [v.strip() for v in values]
            values = [v for v in values if v]
            if len(values) == 0:
                # then maybe removed too much
                values = values0
            if values:
                if cli_mode:
                    response_text = '\n\nFINAL ANSWER:\n\n' + values[-1] + '\n\n'
                else:
                    response_text = values[-1]
                chat_history.append({"role": "assistant", "content": response_text})
                yield {"role": "assistant", "content": response_text, "streaming": True, "chat_history": chat_history,
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


def get_turn_title(response_text):
    tag = 'turn_title'
    pattern = fr'<{tag}>(.*?)</{tag}>'
    values = re.findall(pattern, response_text, re.DOTALL)
    values0 = values.copy()
    values = [v.strip() for v in values]
    values = [v for v in values if v]
    if len(values) == 0:
        # then maybe removed too much
        values = values0
    if values:
        turn_title = values[-1]
    else:
        turn_title = response_text[:15] + '...'
    return turn_title


def get_defaults():
    system_prompt = """Let us play a game of "take only the most minuscule step toward the solution."
    <thinking_game>
    * The assistant's text output must be only the very next possible step.
    * Use your text output as a scratch pad in addition to a literal output of some next step.
    * Everytime you make a major shift in thinking, output your high-level current thinking in <thinking> </thinking> XML tags.
    * You should present your response in a way that iterates on that scratch pad space with surrounding textual context.
    * You win the game is you are able to take the smallest text steps possible while still (on average) heading towards the solution.
    * Backtracking is allowed, and generating python code is allowed (but will not be executed, but can be used to think), just on average over many text output turns you must head towards the answer.
    * You must think using first principles, and ensure you identify inconsistencies, errors, etc.
    * You MUST always end with a very brief natural language title (it should just describe the analysis, do not give step numbers) of what you did inside <turn_title> </turn_title> XML tags.  Only a single title is allowed.
    </thinking_game>
    Are you ready to win the game?"""

    # initial_prompt = "Let's solve a complex math problem: Find the roots of the equation x^3 - 6x^2 + 11x - 6 = 0"

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

    NUM_TURNS = int(os.getenv('NUM_TURNS', '10'))  # Number of turns before pausing for continuation
    num_turns_final_mod = NUM_TURNS - 1  # not required, just ok value.  Could be randomized.

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
            NUM_TURNS, show_next, final_prompt,
            temperature, max_tokens,
            num_turns_final_mod,
            show_cot,
            verbose)


def go():
    (model, system_prompt, initial_prompt, expected_answer,
     next_prompts, num_turns, show_next, final_prompt,
     temperature, max_tokens, num_turns_final_mod,
     show_cot, verbose) = get_defaults()
    args = parse_arguments(model, system_prompt, next_prompts, num_turns, show_next, final_prompt,
                           num_turns_final_mod, show_cot, verbose)

    if args.initial_prompt == '':
        initial_prompt_query = input("Enter the initial prompt (hitting enter will use default initial_prompt)\n\n")
        if initial_prompt_query not in ['', '\n', '\r\n']:
            initial_prompt_chosen = initial_prompt_query
        else:
            initial_prompt_chosen = initial_prompt
    else:
        initial_prompt_chosen = args.initial_prompt

    generator = manage_conversation(model=args.model,
                                    system=args.system_prompt,
                                    initial_prompt=initial_prompt_chosen,
                                    next_prompts=args.next_prompts,
                                    final_prompt=args.final_prompt,
                                    num_turns_final_mod=args.num_turns_final_mod,
                                    num_turns=args.num_turns,
                                    temperature=args.temperature,
                                    max_tokens=args.max_tokens,
                                    seed=args.seed,
                                    cli_mode=True)
    response = ''
    conversation_history = []

    try:
        step = 1
        while True:
            chunk = next(generator)
            if 'role' in chunk and chunk['role'] == 'assistant':
                response += chunk['content']

                if 'turn_title' in chunk and chunk['turn_title']:
                    step_time = f' in time {str(int(chunk["thinking_time"]))}s'
                    acum_time = f' in total {str(int(chunk["total_thinking_time"]))}s'
                    extra = '\n\n' if show_cot else ''
                    extra2 = '**' if show_cot else ''
                    extra3 = ' ' if show_cot else ''
                    print(
                        f'{extra}{extra2}{extra3}Completed Step {step}: {chunk["content"]}{step_time}{acum_time}{extra3}{extra2}{extra}')
                    step += 1
                elif 'final' in chunk and chunk['final']:
                    if '\n' in chunk['content'] or '\r' in chunk['content']:
                        print(f'\n\nFinal Answer:\n\n {chunk["content"]}')
                    else:
                        print('\n\nFinal Answer:\n\n**', chunk['content'], '**\n\n')
                elif show_cot:
                    print(chunk['content'], end='')
                if 'chat_history' in chunk:
                    conversation_history = chunk['chat_history']
            elif 'role' in chunk and chunk['role'] == 'user':
                if not chunk['initial'] and not show_next:
                    if show_cot:
                        print('\n\n')
                    continue
                print('\n', end='')  # finish assistant
                print('\nUser: ', chunk['content'], end='\n\n')
                print('\nAssistant:\n\n ')
            time.sleep(0.001)
    except StopIteration as e:
        pass

    if verbose:
        print("Conversation history:", conversation_history)

    if expected_answer and expected_answer in conversation_history[-1]['content']:
        print("\n\nGot Expected answer!")

    if not show_cot:
        print("**FULL RESPONSE:**\n\n")
        print(response)
    return response


if __name__ == '__main__':
    go()
