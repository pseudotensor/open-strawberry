import os
from typing import List, Dict

# https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
import anthropic

clawd_key = os.getenv('ANTHROPIC_API_KEY')
clawd_client = anthropic.Anthropic(api_key=clawd_key) if clawd_key else None


def get_anthropic(model: str, prompt: str, temperature: float = 0, system: str = ''):
    message = clawd_client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        system=system,
    )
    return message.content[0].text


# Also applies to ollama, vLLM, h2oGPT, etc.
from openai import OpenAI

openai_key = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=openai_key) if openai_key else None


def get_openai(model: str, prompt: str, temperature: float = 0, system: str = ''):
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt}]
    responses = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return responses.choices[0].message.content


# https://github.com/google-gemini/cookbook/
# https://ai.google.dev/gemini-api/docs/caching?lang=python
import google.generativeai as genai

gemini_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_key) if gemini_key else None


def get_gemini(model: str, prompt: str, temperature: float = 0, system: str = ''):
    model = genai.GenerativeModel(model, system_instruction=system, generation_config={'temperature': temperature})
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text


system_prompt = """Let us play a game of "take only the most minuscule step toward the solution."
<thinking_game>
* The assistant's text output must be only the very next possible step.
* Use your text output as a scratch pad in addition to a literal output of some next step.
* Everytime you make a major shift in thinking, output your high-level current thiking in <thinking> </thinking> XML tags.
* You should present your response in a way that iterates on that scratch pad space with surrounding textual context.
* You win the game is you are able to take the smallest text steps possible while still (on average) heading towards the solution.
* Backtracking is allowed, and generating python code is allowed (but will not be executed, but can be used to think), just on average over many text output turns you must head towards the answer.
* You should think like a human, and ensure you identify inconsistencies, errors, etc.
</thinking_game>
Are you ready to win the game?"""
