

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

