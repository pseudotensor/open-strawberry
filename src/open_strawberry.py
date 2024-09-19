import os
import random
import time
from typing import List, Dict, Generator, Tuple
from collections import deque

from src.models import get_model_api
from src.utils import get_turn_title, get_final_answer, get_xml_tag_value


class DeductionTracker:
    def __init__(self):
        self.deductions = []
        self.certainty_scores = []

    def add_deduction(self, deduction: str, certainty: float):
        self.deductions.append(deduction)
        self.certainty_scores.append(certainty)

    def get_deductions(self):
        return list(zip(self.deductions, self.certainty_scores))

    def update_certainty(self, index: int, new_certainty: float):
        if 0 <= index < len(self.certainty_scores):
            self.certainty_scores[index] = new_certainty


class ProblemRepresentation:
    def __init__(self):
        self.current_representation = ""

    def update(self, new_representation: str):
        self.current_representation = new_representation

    def get(self) -> str:
        return self.current_representation


def get_last_assistant_responses(chat_history, n=3):
    assistant_messages = [msg['content'] for msg in chat_history if msg['role'] == 'assistant']
    return assistant_messages[-n:]


def generate_dynamic_system_prompt(base_prompt, turn_count: int, problem_complexity: float, problem_representation: str,
                                   deductions: List[Tuple[str, float]]) -> str:
    dynamic_prompt = base_prompt + "\n\n* Always refer to and update the current problem representation as needed."
    dynamic_prompt += "\n* Maintain and update the list of deductions and their certainty scores."

    if turn_count > 20:
        dynamic_prompt += "\n* At this stage, focus on synthesizing your previous thoughts and looking for breakthrough insights."

    if problem_complexity > 0.7:
        dynamic_prompt += "\n* This is a highly complex problem. Consider breaking it down into smaller subproblems and solving them incrementally."

    dynamic_prompt += "\n* Regularly verify that your current understanding satisfies ALL given clues."
    dynamic_prompt += "\n* If you reach a contradiction, backtrack to the last point where you were certain and explore alternative paths."

    dynamic_prompt += f"\n\nCurrent problem representation:\n{problem_representation}"
    dynamic_prompt += "\n\nYou can update this representation by providing a new one within <representation></representation> tags."

    dynamic_prompt += "\n\nCurrent deductions and certainty scores:"
    for deduction, certainty in deductions:
        dynamic_prompt += f"\n- {deduction} (Certainty: {certainty})"
    dynamic_prompt += "\n\nYou can add new deductions or update existing ones using <deduction></deduction> and <certainty></certainty> tags."

    return dynamic_prompt


def generate_initial_representation_prompt(initial_prompt: str) -> str:
    return f"""Based on the following problem description:
{initial_prompt}

Representation:
* Create a clear and clean representation that breaks down the problem into parts in order to create a structure that helps track solving the problem.
* Put the representation inside <representation> </representation> XML tags, ensuring to add new lines before and after XML tags.
* The representation could be a table, matrix, grid, or any other format that breaks down the problem into its components and ensure it helps iteratively track progress towards the solution.
* Example representations include a matrix that has values of digits as rows and position of digits as columns and values at each row-column as tracking confirmed position, eliminated position, or possible position.
* For a table or grid representation, you must put the table or grid inside a Markdown code block (with new lines around the back ticks) and make it nice and easy to read for a human to understand it.

Deductions:
* Provide your initial deductions (if any) using <deduction></deduction> tags, each followed by a certainty score in <certainty></certainty> tags (0-100).
"""


def generate_verification_prompt(chat_history: List[Dict], turn_count: int, problem_representation: str,
                                 deductions: List[Tuple[str, float]]) -> str:
    last_responses = get_last_assistant_responses(chat_history, n=5)

    verification_prompt = f"""Turn {turn_count}: Comprehensive Verification and Critique

1. Review your previous reasoning steps:
{' '.join(last_responses)}

2. Current problem representation:
{problem_representation}

3. Current deductions and certainty scores:
{deductions}

4. Perform the following checks:
   a) Identify any logical fallacies or unjustified assumptions
   b) Check for mathematical or factual errors
   c) Assess the relevance of each step to the main problem
   d) Evaluate the coherence and consistency of your reasoning
   e) Verify that your current understanding satisfies ALL given clues
   f) Check if any of your deductions contradict each other

5. If you find any issues:
   a) Explain the issue in detail
   b) Correct the error or resolve the contradiction
   c) Update the problem representation if necessary
   d) Update deductions and certainty scores as needed

6. If no issues are found, suggest a new approach or perspective to consider.

7. Assign an overall confidence score (0-100) to your current reasoning path and explain why.

Respond in this format:
<verification>
[Your detailed verification and critique]
</verification>
<updates>
[Any updates to the problem representation or deductions]
</updates>
<confidence_score>[0-100]</confidence_score>
<explanation>[Explanation for the confidence score]</explanation>

If you need to update the problem representation, provide the new representation within <representation></representation> tags.
For new or updated deductions, use <deduction></deduction> tags, each followed by <certainty></certainty> tags.
"""
    return verification_prompt


def generate_hypothesis_prompt(chat_history: List[Dict]) -> str:
    return """Based on your current understanding of the problem:
1. Generate three distinct hypotheses that could lead to a solution.
2. For each hypothesis, provide a brief rationale and a potential test to validate it.
3. Rank these hypotheses in order of perceived promise.

Respond in this format:
<hypotheses>
1. [Hypothesis 1]
   Rationale: [Brief explanation]
   Test: [Proposed validation method]

2. [Hypothesis 2]
   Rationale: [Brief explanation]
   Test: [Proposed validation method]

3. [Hypothesis 3]
   Rationale: [Brief explanation]
   Test: [Proposed validation method]
</hypotheses>
<ranking>[Your ranking and brief justification]</ranking>
"""


def generate_analogical_reasoning_prompt(problem_description: str) -> str:
    return f"""Consider the following problem:
{problem_description}

Now, think of an analogous problem from a different domain that shares similar structural characteristics. Describe:
1. The analogous problem
2. The key similarities between the original and analogous problems
3. How the solution to the analogous problem might inform our approach to the original problem

Respond in this format:
<analogy>
Problem: [Description of the analogous problem]
Similarities: [Key structural similarities]
Insights: [How this analogy might help solve the original problem]
</analogy>
"""


def generate_metacognitive_prompt() -> str:
    return """Take a step back and reflect on your problem-solving process:
1. What strategies have been most effective so far?
2. What are the main obstacles you're facing?
3. How might you adjust your approach to overcome these obstacles?
4. What assumptions might you be making that could be limiting your progress?

Respond in this format:
<metacognition>
Effective Strategies: [List and brief explanation]
Main Obstacles: [List and brief explanation]
Proposed Adjustments: [List of potential changes to your approach]
Potential Limiting Assumptions: [List and brief explanation]
</metacognition>
"""


def generate_devils_advocate_prompt(current_approach: str) -> str:
    return f"""Consider your current approach:
{current_approach}

Now, play the role of a skeptical critic:
1. What are the three strongest arguments against this approach?
2. What critical information might we be overlooking?
3. How might this approach fail in extreme or edge cases?

Respond in this format:
<devils_advocate>
Counter-arguments:
1. [First strong counter-argument]
2. [Second strong counter-argument]
3. [Third strong counter-argument]

Overlooked Information: [Potential critical information we might be missing]

Potential Failures: [How this approach might fail in extreme or edge cases]
</devils_advocate>
"""


def generate_hint(problem_description: str, current_progress: str, difficulty: float) -> str:
    if difficulty < 0.3:
        hint_level = "subtle"
    elif difficulty < 0.7:
        hint_level = "moderate"
    else:
        hint_level = "strong"

    return f"""Based on the original problem:
{problem_description}

And the current progress:
{current_progress}

Provide a {hint_level} hint to help move towards the solution without giving it away entirely.

<hint>
[Your {hint_level} hint here]
</hint>
"""


def summarize_and_restructure(chat_history: List[Dict]) -> str:
    return """Review the entire conversation history and provide:
1. A concise summary of the key insights and progress made so far
2. A restructured presentation of the problem based on our current understanding
3. Identification of any patterns or recurring themes in our problem-solving attempts

Respond in this format:
<summary_and_restructure>
Key Insights: [Bullet point list of main insights]
Restructured Problem: [Revised problem statement based on current understanding]
Patterns/Themes: [Identified patterns or recurring themes in our approach]
</summary_and_restructure>
"""


class Memory:
    def __init__(self, max_size=10):
        self.insights = deque(maxlen=max_size)
        self.mistakes = deque(maxlen=max_size)
        self.dead_ends = deque(maxlen=max_size)

    def add_insight(self, insight: str):
        self.insights.append(insight)

    def add_mistake(self, mistake: str):
        self.mistakes.append(mistake)

    def add_dead_end(self, dead_end: str):
        self.dead_ends.append(dead_end)

    def get_insights(self) -> List[str]:
        return list(self.insights)

    def get_mistakes(self) -> List[str]:
        return list(self.mistakes)

    def get_dead_ends(self) -> List[str]:
        return list(self.dead_ends)


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
                        verbose: bool = False
                        ) -> Generator[Dict, None, list]:
    if seed == 0:
        seed = random.randint(0, 1000000)
    random.seed(seed)
    get_model_func = get_model_api(model)
    chat_history = []
    memory = Memory()
    problem_representation = ProblemRepresentation()
    deduction_tracker = DeductionTracker()

    turn_count = 0
    total_thinking_time = 0
    problem_complexity = 0.5  # Initial estimate, will be dynamically updated

    base_system = system
    while True:
        system = generate_dynamic_system_prompt(base_system, turn_count, problem_complexity,
                                                problem_representation.get(), deduction_tracker.get_deductions())
        trying_final = False

        if turn_count == 0:
            prompt = generate_initial_representation_prompt(initial_prompt)
        elif turn_count % 5 == 0:
            prompt = generate_verification_prompt(chat_history, turn_count, problem_representation.get(),
                                                  deduction_tracker.get_deductions())
        elif turn_count % 7 == 0:
            prompt = generate_hypothesis_prompt(chat_history)
        elif turn_count % 11 == 0:
            prompt = generate_analogical_reasoning_prompt(initial_prompt)
        elif turn_count % 13 == 0:
            prompt = generate_metacognitive_prompt()
        elif turn_count % 17 == 0:
            current_approach = get_last_assistant_responses(chat_history, n=1)[0]
            prompt = generate_devils_advocate_prompt(current_approach)
        elif turn_count % 19 == 0:
            current_progress = get_last_assistant_responses(chat_history, n=3)
            prompt = generate_hint(initial_prompt, "\n".join(current_progress), problem_complexity)
        elif turn_count % 23 == 0:
            prompt = summarize_and_restructure(chat_history)
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

        # Update problem complexity based on thinking time and response length
        problem_complexity = min(1.0,
                                 problem_complexity + (thinking_time / 60) * 0.1 + (len(response_text) / 1000) * 0.05)

        # Extract and update problem representation
        representations = get_xml_tag_value(response_text, 'representation', ret_all=False)
        if representations:
            problem_representation.update(representations[-1])

        # Extract and update deductions
        deductions = get_xml_tag_value(response_text, 'deduction')
        for deduction in deductions:
            certainties = get_xml_tag_value(response_text, 'certainty', ret_all=False)
            if certainties:
                deduction_tracker.add_deduction(deduction, certainties[-1])

        # Extract insights, mistakes, and dead ends from the response
        [memory.add_insight(x) for x in get_xml_tag_value(response_text, 'insight')]
        [memory.add_mistake(x) for x in get_xml_tag_value(response_text, 'mistake')]
        [memory.add_dead_end(x) for x in get_xml_tag_value(response_text, 'dead_end')]

        turn_title = get_turn_title(response_text)
        yield {"role": "assistant", "content": turn_title, "turn_title": True, 'thinking_time': thinking_time,
               'total_thinking_time': total_thinking_time}

        chat_history.append(
            {"role": "user",
             "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]})
        chat_history.append({"role": "assistant", "content": response_text})

        # Adjusted to only check final answer when trying_final is True
        always_check_final = False
        if trying_final or always_check_final:
            final_value = get_final_answer(response_text, cli_mode=cli_mode)
            if final_value:
                chat_history.append({"role": "assistant", "content": final_value})
                yield {"role": "assistant", "content": final_value, "streaming": True, "chat_history": chat_history,
                       "final": True}
                break

        turn_count += 1

        # Dynamically adjust temperature based on progress
        if turn_count % 10 == 0:
            temperature = min(1.0, temperature + 0.1)  # Gradually increase temperature to encourage exploration

        if turn_count % num_turns == 0:
            # periodically pause for continuation, never have to fully terminate
            if cli_mode:
                user_continue = input("\nContinue? (y/n): ").lower() == 'y'
                if not user_continue:
                    break
            else:
                yield {"role": "action", "content": "continue?", "chat_history": chat_history}

        time.sleep(0.001)


def get_defaults() -> Tuple:
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
</thinking_game>"""

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
