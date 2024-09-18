import argparse
import time

from src.open_strawberry import get_defaults, manage_conversation


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


def go_cli():
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
