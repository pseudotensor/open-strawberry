import ast
import datetime
import os
from typing import List, Dict, Generator
from dotenv import load_dotenv

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# Load environment variables from .env file
load_dotenv()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def anthropic_completion_with_backoff(client, *args, **kwargs):
    return client.beta.prompt_caching.messages.create(*args, **kwargs)


def get_anthropic(model: str,
                  prompt: str,
                  temperature: float = 0,
                  max_tokens: int = 4096,
                  system: str = '',
                  chat_history: List[Dict] = None,
                  secrets: Dict = {},
                  verbose=False) -> \
        Generator[dict, None, None]:
    model = model.replace('anthropic:', '')

    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    import anthropic

    clawd_key = secrets.get('ANTHROPIC_API_KEY')
    clawd_client = anthropic.Anthropic(api_key=clawd_key) if clawd_key else None

    if chat_history is None:
        chat_history = []

    messages = []

    # Add conversation history, removing cache_control from all but the last two user messages
    for i, message in enumerate(chat_history):
        if message["role"] == "user":
            content = message["content"][0]["text"] if isinstance(message["content"], list) else message["content"]

            if i >= len(chat_history) - 3:  # Last two user messages
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })

            else:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": content}]
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

    response = anthropic_completion_with_backoff(clawd_client,
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


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def openai_completion_with_backoff(client, *args, **kwargs):
    return client.chat.completions.create(*args, **kwargs)


def get_openai(model: str,
               prompt: str,
               temperature: float = 0,
               max_tokens: int = 4096,
               system: str = '',
               chat_history: List[Dict] = None,
               secrets: Dict = {},
               verbose=False) -> Generator[dict, None, None]:
    if model.startswith('ollama:'):
        model = model.replace('ollama:', '')
        openai_key = secrets.get('OLLAMA_OPENAI_API_KEY')
        openai_base_url = secrets.get('OLLAMA_OPENAI_BASE_URL', 'http://localhost:11434/v1/')
    else:
        model = model.replace('openai:', '')
        openai_key = secrets.get('OPENAI_API_KEY')
        openai_base_url = secrets.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')

    from openai import OpenAI

    openai_client = OpenAI(api_key=openai_key, base_url=openai_base_url) if openai_key else None

    if chat_history is None:
        chat_history = []
    chat_history_copy = chat_history.copy()
    for mi, message in enumerate(chat_history_copy):
        if isinstance(message["content"], list):
            chat_history_copy[mi]["content"] = message["content"][0]["text"]
    chat_history = chat_history_copy

    messages = [{"role": "system", "content": system}] + chat_history + [{"role": "user", "content": prompt}]

    response = openai_completion_with_backoff(openai_client,
                                              model=model,
                                              messages=messages,
                                              temperature=temperature,
                                              max_tokens=max_tokens,
                                              stream=True,
                                              )

    output_tokens = 0
    input_tokens = 0
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield dict(text=chunk.choices[0].delta.content)
        if chunk.usage:
            output_tokens = chunk.usage.completion_tokens
            input_tokens = chunk.usage.prompt_tokens

    if verbose:
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens: {input_tokens}")
    yield dict(output_tokens=output_tokens, input_tokens=input_tokens)


def openai_messages_to_gemini_history(messages):
    """Converts OpenAI messages to Gemini history format.

    Args:
        messages: A list of OpenAI messages, each with "role" and "content" keys.

    Returns:
        A list of dictionaries representing the chat history for Gemini.
    """
    history = []
    for message in messages:
        if isinstance(message["content"], list):
            message["content"] = message["content"][0]["text"]
        if message["role"] == "user":
            history.append({"role": "user", "parts": [{"text": message["content"]}]})
        elif message["role"] == "assistant":
            history.append({"role": "model", "parts": [{"text": message["content"]}]})
        # Optionally handle system messages if needed
        # elif message["role"] == "system":
        #     history.append({"role": "system", "parts": [{"text": message["content"]}]})

    return history


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def gemini_send_message_with_backoff(chat, prompt, stream=True):
    return chat.send_message(prompt, stream=stream)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def gemini_generate_content_with_backoff(model, prompt, stream=True):
    return model.generate_content(prompt, stream=stream)


def get_google(model: str,
               prompt: str,
               temperature: float = 0,
               max_tokens: int = 4096,
               system: str = '',
               chat_history: List[Dict] = None,
               secrets: Dict = {},
               verbose=False) -> Generator[dict, None, None]:
    model = model.replace('google:', '').replace('gemini:', '')

    import google.generativeai as genai

    gemini_key = secrets.get("GEMINI_API_KEY")
    genai.configure(api_key=gemini_key)
    # Create the model
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }

    if chat_history is None:
        chat_history = []

    chat_history = chat_history.copy()
    chat_history = openai_messages_to_gemini_history(chat_history)

    # NOTE: assume want own control.  Too many false positives by Google.
    from google.generativeai.types import HarmCategory
    from google.generativeai.types import HarmBlockThreshold
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    cache = None
    # disable cache for now until work into things well
    use_cache = False
    if use_cache and model == 'gemini-1.5-pro':
        from google.generativeai import caching
        # Estimate token count (this is a rough estimate, you may need a more accurate method)
        estimated_tokens = len(prompt.split()) + sum(len(msg['content'].split()) for msg in chat_history)

        if estimated_tokens > 32000:
            cache = caching.CachedContent.create(
                model=model,
                display_name=f'cache_{datetime.datetime.now().isoformat()}',
                system_instruction=system,
                contents=[prompt] + [msg['content'] for msg in chat_history],
                ttl=datetime.timedelta(minutes=5),  # Set an appropriate TTL.  Short for now for cost savings.
            )
            gemini_model = genai.GenerativeModel.from_cached_content(cached_content=cache)
        else:
            gemini_model = genai.GenerativeModel(model_name=model,
                                                 generation_config=generation_config,
                                                 safety_settings=safety_settings)
    else:
        gemini_model = genai.GenerativeModel(model_name=model,
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)

    if cache:
        response = gemini_generate_content_with_backoff(gemini_model, prompt, stream=True)
    else:
        chat = gemini_model.start_chat(history=chat_history)
        response = gemini_send_message_with_backoff(chat, prompt, stream=True)

    output_tokens = 0
    input_tokens = 0
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0

    for chunk in response:
        if chunk.text:
            yield dict(text=chunk.text)
        if chunk.usage_metadata:
            output_tokens = chunk.usage_metadata.candidates_token_count
            input_tokens = chunk.usage_metadata.prompt_token_count
            cache_read_input_tokens = chunk.usage_metadata.cached_content_token_count
            cache_creation_input_tokens = 0  # This might need to be updated if available in the API

    if verbose:
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens: {input_tokens}")
        print(f"Cached tokens: {cache_read_input_tokens}")

    yield dict(output_tokens=output_tokens, input_tokens=input_tokens,
               cache_read_input_tokens=cache_read_input_tokens,
               cache_creation_input_tokens=cache_creation_input_tokens)


def delete_cache(cache):
    if cache:
        cache.delete()
        print(f"Cache {cache.display_name} deleted.")
    else:
        print("No cache to delete.")


def get_groq(model: str,
             prompt: str,
             temperature: float = 0,
             max_tokens: int = 4096,
             system: str = '',
             chat_history: List[Dict] = None,
             secrets: Dict = {},
             verbose=False) -> Generator[dict, None, None]:
    model = model.replace('groq:', '')

    from groq import Groq

    groq_key = secrets.get("GROQ_API_KEY")
    client = Groq(api_key=groq_key)

    if chat_history is None:
        chat_history = []

    chat_history = chat_history.copy()

    messages = [{"role": "system", "content": system}] + chat_history + [{"role": "user", "content": prompt}]

    stream = openai_completion_with_backoff(client,
                                            messages=messages,
                                            model=model,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            stream=True,
                                            )

    output_tokens = 0
    input_tokens = 0
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield dict(text=chunk.choices[0].delta.content)
        if chunk.usage:
            output_tokens = chunk.usage.completion_tokens
            input_tokens = chunk.usage.prompt_tokens

    if verbose:
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens: {input_tokens}")
    yield dict(output_tokens=output_tokens, input_tokens=input_tokens)


def get_cerebras(model: str,
                 prompt: str,
                 temperature: float = 0,
                 max_tokens: int = 4096,
                 system: str = '',
                 chat_history: List[Dict] = None,
                 secrets: Dict = {},
                 verbose=False) -> Generator[dict, None, None]:
    # context_length is only 8207
    model = model.replace('cerebras:', '')

    from cerebras.cloud.sdk import Cerebras

    api_key = secrets.get("CEREBRAS_OPENAI_API_KEY")
    client = Cerebras(api_key=api_key)

    if chat_history is None:
        chat_history = []

    chat_history = chat_history.copy()

    messages = [{"role": "system", "content": system}] + chat_history + [{"role": "user", "content": prompt}]

    stream = openai_completion_with_backoff(client,
                                            messages=messages,
                                            model=model,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            stream=True,
                                            )

    output_tokens = 0
    input_tokens = 0
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield dict(text=chunk.choices[0].delta.content)
        if chunk.usage:
            output_tokens = chunk.usage.completion_tokens
            input_tokens = chunk.usage.prompt_tokens

    if verbose:
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens: {input_tokens}")
    yield dict(output_tokens=output_tokens, input_tokens=input_tokens)


def get_openai_azure(model: str,
                     prompt: str,
                     temperature: float = 0,
                     max_tokens: int = 4096,
                     system: str = '',
                     chat_history: List[Dict] = None,
                     secrets: Dict = {},
                     verbose=False) -> Generator[dict, None, None]:
    model = model.replace('azure:', '').replace('openai_azure:', '')

    from openai import AzureOpenAI

    azure_endpoint = secrets.get("AZURE_OPENAI_ENDPOINT")  # e.g. https://project.openai.azure.com
    azure_key = secrets.get("AZURE_OPENAI_API_KEY")
    azure_deployment = secrets.get("AZURE_OPENAI_DEPLOYMENT")  # i.e. deployment name with some models deployed
    azure_api_version = secrets.get('AZURE_OPENAI_API_VERSION', '2024-07-01-preview')
    assert azure_endpoint is not None, "Azure OpenAI endpoint not set"
    assert azure_key is not None, "Azure OpenAI API key not set"
    assert azure_deployment is not None, "Azure OpenAI deployment not set"

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        api_version=azure_api_version,
        azure_deployment=azure_deployment,
    )

    if chat_history is None:
        chat_history = []

    messages = [{"role": "system", "content": system}] + chat_history + [{"role": "user", "content": prompt}]

    response = openai_completion_with_backoff(client,
                                              model=model,
                                              messages=messages,
                                              temperature=temperature,
                                              max_tokens=max_tokens,
                                              stream=True
                                              )

    output_tokens = 0
    input_tokens = 0
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield dict(text=chunk.choices[0].delta.content)
        if chunk.usage:
            output_tokens = chunk.usage.completion_tokens
            input_tokens = chunk.usage.prompt_tokens

    if verbose:
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens: {input_tokens}")
    yield dict(output_tokens=output_tokens, input_tokens=input_tokens)


def to_list(x):
    if x:
        try:
            ollama_model_list = ast.literal_eval(x)
            assert isinstance(ollama_model_list, list)
        except:
            x = [x]
    else:
        x = []
    return x


def get_model_names(secrets, on_hf_spaces=False):
    if not on_hf_spaces:
        secrets = os.environ
    if secrets.get('ANTHROPIC_API_KEY'):
        anthropic_models = ['claude-3-5-sonnet-20240620', 'claude-3-haiku-20240307', 'claude-3-opus-20240229']
    else:
        anthropic_models = []
    if secrets.get('OPENAI_API_KEY'):
        if secrets.get('OPENAI_MODEL_NAME'):
            openai_models = to_list(secrets.get('OPENAI_MODEL_NAME'))
        else:
            openai_models = ['gpt-4o', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini']
    else:
        openai_models = []
    if secrets.get('AZURE_OPENAI_API_KEY'):
        if secrets.get('AZURE_OPENAI_MODEL_NAME'):
            azure_models = to_list(secrets.get('AZURE_OPENAI_MODEL_NAME'))
        else:
            azure_models = ['gpt-4o', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini']
    else:
        azure_models = []
    if secrets.get('GEMINI_API_KEY'):
        google_models = ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest']
    else:
        google_models = []
    if secrets.get('GROQ_API_KEY'):
        groq_models = ['llama-3.1-70b-versatile',
                       'llama-3.1-8b-instant',
                       'llama3-groq-70b-8192-tool-use-preview',
                       'llama3-groq-8b-8192-tool-use-preview',
                       'mixtral-8x7b-32768']
    else:
        groq_models = []
    if secrets.get('CEREBRAS_OPENAI_API_KEY'):
        cerebras_models = ['llama3.1-70b', 'llama3.1-8b']
    else:
        cerebras_models = []
    if secrets.get('OLLAMA_OPENAI_API_KEY'):
        ollama_model = os.environ['OLLAMA_OPENAI_MODEL_NAME']
        ollama_model = to_list(ollama_model)
    else:
        ollama_model = []

    groq_models = ['groq:' + x for x in groq_models]
    cerebras_models = ['cerebras:' + x for x in cerebras_models]
    azure_models = ['azure:' + x for x in azure_models]
    openai_models = ['openai:' + x for x in openai_models]
    google_models = ['google:' + x for x in google_models]
    anthropic_models = ['anthropic:' + x for x in anthropic_models]
    ollama = ['ollama:' + x if 'ollama:' not in x else x for x in ollama_model]

    return anthropic_models + openai_models + google_models + groq_models + cerebras_models + azure_models + ollama


def get_model_api(model: str):
    assert model not in ['', None], "Model not set, need to add API key to have models appear and select one."
    if model.startswith('anthropic:'):
        return get_anthropic
    elif model.startswith('openai:') or model.startswith('ollama:'):
        return get_openai
    elif model.startswith('google:'):
        return get_google
    elif model.startswith('groq:'):
        return get_groq
    elif model.startswith('cerebras:'):
        return get_cerebras
    elif model.startswith('azure:'):
        return get_openai_azure
    else:
        raise ValueError(
            f"Unsupported model: {model}.  Ensure to add prefix (e.g. openai:, google:, groq:, cerebras:, azure:, ollama:, anthropic:)")
