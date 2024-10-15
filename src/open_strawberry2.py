import ast
import os
import random
import time
from typing import List, Dict, Generator, Tuple

try:
    from src.models import get_model_api
    from src.utils import get_turn_title, get_final_answer, get_xml_tag_value
except (ModuleNotFoundError, ImportError):
    from models import get_model_api
    from utils import get_turn_title, get_final_answer, get_xml_tag_value


# https://arxiv.org/abs/2401.10020
def get_verification_system_prompt(context):
    verification_system_prompt = f"""You are a critical analyzer and fact-checker. Your task is to carefully examine the given response and provide a thorough critique. Focus on the following aspects:

<definitions>
* Clarity: Is the response clear and unambiguous?
* Relevance: Does the response contribute to solving the problem at hand?
* Consistency: Does the response align with previously stated information?
* Logical Soundness: Is the reasoning in the response valid?
* Factual Accuracy: Are any claims in the response accurate (if applicable)?
</definitions>

<scoring_rubric>
* Add 1 point if the response is helpful, even if it is incomplete or contains some irrelevant content or may not be entirely logical or accurate.
* Add another point if the response is relevant, but may lack clarity or logical soundness.
* Award a third point if the response is helpful, regardless of being incomplete or slightly off-topic.
* Grant a fourth point if the response is clearly written and is fully comprehensively and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
* Bestow a fifth point for a response that is impeccable, without extraneous information, perfectly consistent, exactly logical, and reflects expert knowledge.
* Deduct 1 point for a major clarity issue
* Deduct 1 point for a major relevance issue
* Deduct 1 point for a major consistency issue
* Deduct 1 point for a major logical fallacy issue
* Deduct 1 point for a major factual accuracy issue
</scoring_rubric>
These points are additive in nature, so one can get up to 5 points for a perfect response and as low as -5 points for a completely poor response.

Prior context that led to the text to split:
<context>
{context}
</context>

After examining the user's instruction and the response:
* Provide a brief 100-word analysis addressing these aspects. If you identify any issues or inconsistencies, explain them clearly.  You MUST provide a critique in <critique> </critique> XML tags.
* Conclude with a purely numerical cumulative score.  You MUST provide a score inside <score </score> XML tags (e.g., <score>3</score> for 3 points).
"""
    return verification_system_prompt


def sentence_splitting_system_prompt(context) -> str:
    return f"""You are a sentence splitter.
Your task is to split the given text into sentences.
* Ensure that each sentence is properly formatted and correctly separated.
* Use proper punctuation and capitalization.
* If a sentence is ambiguous or unclear, use your best judgment to split it into two or more sentences.
* If a sentence is already correctly formatted, do not make any changes.
* You may also need to correct any spelling or grammatical errors in the text.
* Remove any unnecessary line breaks.
* Omit needless sentences and words that are not directly relevant to the context, i.e. avoid conversation fillers or discourse markers.

Prior context that led to the text to split:
<context>
{context}
</context>

For your response, each output sentence should be separated by a single newline character.
Your sentences should be all contained within <sentences> </sentences> XML tags.
"""


def split_into_sentences(text: str, context: str, get_model_func, model, secrets, verbose=False) -> List[str]:
    return [text]


def split_into_sentences_001(text: str, context: str, get_model_func, model, secrets, verbose=False) -> List[str]:
    sentences = ''
    for chunk in get_model_func(model, text, system=sentence_splitting_system_prompt(context),
                                temperature=0, max_tokens=4096,
                                secrets=secrets, verbose=verbose):
        if 'text' in chunk and chunk['text']:
            sentences += chunk['text']
    sentences = get_xml_tag_value(sentences, 'sentences')
    if sentences:
        sentences = sentences[0]
    return sentences.split('\n')


def manage_conversation(model: str,
                        system: str,
                        initial_prompt: str,
                        next_prompts: List[str],
                        final_prompt: str = "",
                        num_turns: int = 25,
                        num_turns_final_mod: int = 9,
                        cli_mode: bool = False,
                        temperature: float = 0.3,
                        max_tokens: int = 4096,
                        seed: int = 1234,
                        secrets: Dict = {},
                        verbose: bool = False,
                        ) -> Generator[Dict, None, list]:
    if seed == 0:
        seed = random.randint(0, 1000000)
    random.seed(seed)
    get_model_func = get_model_api(model)
    chat_history = []

    turn_count = 0
    total_thinking_time = 0
    response_text = ''

    # Initial prompt
    yield {"role": "user", "content": initial_prompt, "chat_history": chat_history, "initial": True}

    while True:
        thinking_time = time.time()

        # Determine the current prompt and system
        if turn_count == 0:
            current_prompt = initial_prompt
        elif turn_count % num_turns_final_mod == 0 and final_prompt:
            current_prompt = final_prompt
        else:
            # default if nothing to verify
            current_prompt = next_prompts[0]

            # Verification step
            context = chat_history[-2]['content'] if len(chat_history) >= 2 else initial_prompt
            if context:
                sentences = split_into_sentences(response_text, context, get_model_func, model, secrets, verbose=verbose)
                verification_context = "\n".join([msg["content"] for msg in chat_history[-1:] if isinstance(msg["content"], str)])
                verification_responses = []

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sentence_verification = ''
                    for chunk in get_model_func(model, sentence,
                                                system=get_verification_system_prompt(verification_context),
                                                temperature=temperature, max_tokens=max_tokens,
                                                secrets=secrets, verbose=verbose):
                        if 'text' in chunk and chunk['text']:
                            sentence_verification += chunk['text']
                        else:
                            yield {"role": "usage", "content": chunk}

                    valid_score = get_xml_tag_value(sentence_verification, 'score')
                    if valid_score:
                        valid_score = valid_score[0]
                    try:
                        valid_score = ast.literal_eval(valid_score)
                    except Exception as e:
                        print(f"Error parsing score: {valid_score} Error: {e}")
                        # if invalid validation, skip
                        valid_score = 0

                    critique = get_xml_tag_value(sentence_verification, 'critique')
                    if critique:
                        critique = critique[0]

                    # valid_score_threshold = 0
                    valid_score_threshold = 5
                    if valid_score < valid_score_threshold:
                        verification_responses.append(f"Critique for sentence: '{sentence}'\n{critique}\n")

                current_prompt = "\n".join(verification_responses)
                if not current_prompt:
                    # current_prompt = next_prompts[turn_count % len(next_prompts)]
                    current_prompt = next_prompts[0]

        # Main LLM call, always responding to verification responses
        response_text = ''
        for chunk in get_model_func(model, current_prompt, system=system, chat_history=chat_history,
                                    temperature=temperature, max_tokens=max_tokens,
                                    secrets=secrets, verbose=verbose):
            if 'text' in chunk and chunk['text']:
                response_text += chunk['text']
                yield {"role": "assistant",
                       "content": chunk['text'],
                       "streaming": True,
                       "chat_history": chat_history,
                       "final": False,
                       "turn_title": False}
            else:
                yield {"role": "usage", "content": chunk}

        thinking_time = time.time() - thinking_time
        total_thinking_time += thinking_time

        if turn_count == 0:
            turn_title = get_turn_title(response_text)
            yield {"role": "assistant", "content": turn_title, "turn_title": True, 'thinking_time': thinking_time,
                   'total_thinking_time': total_thinking_time}

        chat_history.append({"role": "user", "content": current_prompt})

        # Check for final answer
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

        # reduce output down to only necessary step
        thinking = get_xml_tag_value(response_text, 'thinking')
        if thinking:
            thinking = thinking[0]
        else:
            thinking = ''
        step_text = get_xml_tag_value(response_text, 'step')
        if step_text:
            step_text = step_text[0]
        else:
            step_text = ''
        response_text_tmp = f"{thinking}\n{step_text}".strip()
        chat_history.append({"role": "assistant", "content": response_text_tmp if response_text_tmp else response_text})
        response_text = response_text_tmp

        time.sleep(0.001)


def get_defaults() -> Tuple:
    on_hf_spaces = os.getenv("HF_SPACES", '0') == '1'
    if on_hf_spaces:
        initial_prompt = "How many r's are in strawberry?"
        expected_answer = "3"
    else:

        initial_prompt = """Can you crack the code?
    9 2 8 5 (One number is correct but in the wrong position)
    1 9 3 7 (Two numbers are correct but in the wrong positions)
    5 2 0 1 (one number is correct and in the right position)
    6 5 0 7 (nothing is correct)
    8 5 2 4 (two numbers are correct but in the wrong positions)"""

        expected_answer = "3841"

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
* If you have reduced the possible solutions to a small set, try to carefully verify each one before giving your final answer as all of them.
* Try to find a useful representation of the problem and that is human interpretable, e.g. a table, list, or map.
</thinking_game>
Remember to compensate for your flaws:
<system_flaws>
* Flaw 1: Bad at counting due to tokenization issues.  Expand word with spaces between first and then only count that expanded version.
* Flaw 2: Grade school or advanced math.  Solve such problems very carefully step-by-step.
* Flaw 3: Strict positional information understanding, e.g. tic-tac-toe grid.  Use techniques to ensure you are correctly interpreting positional information highly accurately.
</system_flaws>
Final points:
* You MUST place a brief justification of your step as thoughts <thinking> </thinking> XML tags.
* You MUST place your most minuscule step towards the solution in <step> </step> XML tags.
* Any other general thoughts can be placed outside the thinking and step XML tags.
"""

    next_prompts = [
        "Continue your effort to answer the original query. What's your next step?",
        "What aspect of the problem haven't we considered yet?",
        "Can you identify any patterns or relationships in the given information?",
        "How would you verify your current reasoning?",
        "What's the weakest part of your current approach? How can we strengthen it?",
        "If you were to explain your current thinking to a novice, what would you say?",
        "What alternative perspectives could we consider?",
        "How does your current approach align with the constraints of the problem?",
        "What assumptions are we making? Are they all necessary?",
        "If we were to start over with our current knowledge, how would our approach differ?",
    ]

    final_prompt = """Verification checklist:
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
    max_tokens = 4096

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
