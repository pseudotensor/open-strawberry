# open-strawberry

![img_0.png](img_0.jpg)

A proof-of-concept to construct of reasoning traces to build an open-source version of OpenAI o1 as inspired by OpenAI's Strawberry algorithm.

If you want to support the project, turn ★ into ⭐ (top-right corner) and share it with your friends.

Contributions very welcome!

![img.png](img.png)

## Installation

Python >=3.10 should be fine, then:
```bash
pip install -r requirements.txt
```

## Usage

Fill `.env` with required API keys etc or set ENVs, e.g.:
```.env
# OpenAI
OPENAI_API_KEY=
OPENAI_BASE_URL=

# ollama
OLLAMA_OPENAI_API_KEY=
OLLAMA_OPENAI_BASE_URL=
# quoted list of strings or string
OLLAMA_OPENAI_MODEL_NAME=

# Azure
AZURE_OPENAI_API_KEY=
OPENAI_API_VERSION=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=

# Others
GEMINI_API_KEY=
MISTRAL_API_KEY=
GROQ_API_KEY=
ANTHROPIC_API_KEY=

# WIP: not yet used
HUGGING_FACE_HUB_TOKEN=
REPLICATE_API_TOKEN=
TOGETHERAI_API_TOKEN=
```

### ollama

For ollama, one can use the OpenAI service:
```bash
#Shut down ollama and re-run on whichever GPUs wanted:
sudo systemctl stop ollama.service
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11434 ollama serve &> ollama.log &
ollama run mistral:v0.3
```
then choose set `.env` with `OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1/` and e.g. `OLLAMA_OPENAI_MODEL_NAME=ollama:mistral:v0.3` or list of ollama models: `OLLAMA_OPENAI_MODEL_NAME="[ollama:mistral:v0.3"]`,
then run for CLI:
```bash
python src/open_strawberry.py --model ollama:mistral:v0.3
```
or pick the model in the UI.

Using UI:
```bash
export ANTHROPIC_API_KEY=your_api_key
streamlit run src/app.py
```
then open the browser to http://localhost:8501 (should pop-up automatically).

Using CLI:
```bash
export ANTHROPIC_API_KEY=your_api_key
python src/open_strawberry.py
```
then choose prompt.

The project is in its initial stages to explore generation of reasoning traces for specific problems as proof of concept.

Note that the demo prompt is simple models and even sonnet3.5 and gpt-4o cannot find a solution even with standard CoT.  Only o1-mini or o1-preview can sometimes get, although [code agents](https://x.com/ArnoCandel/status/1834306725706694916) and easily solve it.

## Background

open-strawberry is based on speculations about OpenAI's Strawberry, a refined search-generation algorithm for generating and verifying training data.

This project aims to recreate a similar system using open-source tools and methodologies.

### Speculative Definitions

- **Q***: A hypothetical primordial search-generation algorithm developed by OpenAI to generate training data.
- **Strawberry**: An advanced search-generation algorithm by OpenAI for generating and verifying training data.
- **o1**: GPT-4o and GPT-4o-mini based but fine-tuned on Strawberry data, including o1-mini, o1-preview, o1, and o1-ioi. [1]
- **Orion**: GPT-5-based model that incorporates Strawberry's synthetic data and manages 0-shot vs. long reasoning queries better.

## Generating Reasoning Traces

Bootstrapping is key via progressive learning.

1. Bootstrap starting from existing supervised fine-tuned, instruction-tuned, preference-tuned models using multi-turn chat history.
2. Implement a prompt system that guides the LLM to take incremental steps towards a solution.
3. Randomized useful CoT prompts from user (e.g. not just next but "are you sure?" "any mistakes?" "how would you verify your answer?") to illicit diverse reasoning and introspection.
4. Emphasize the LLM to make the most miniscule step toward the solution, e.g. even a single phrase or sentence is preferred.  Only once the final answer would be produced should an extended full response be given.
5. Generate multi-turn chat reasoning traces
6. Sometimes ask if the model is confident about an answer.  If so, then ask it to place that answer in <final_answer> xml tags.  If done, then terminate the reasoning trace generation.
7. Employ a verification system to check for errors in the chat history.
8. Generate multiple reasoning traces per problem.
9. Apply this process to a large set of problems with verifiable ground truths.
10. Identify problems the existing instruct model can do just barely with strong CoT and high temperature for some number of fixed (e.g. 20) repeats.
 
## Fine-Tuning on Reasoning Traces

1. Select correct and incorrect reasoning traces for each problem based upon the ground truth.
2. Fine-tune a model using the selected reasoning traces using DPO or NLHF, where the preference is positive for correct traces, negative for incorrect traces.
3. Skew the preference weight by number of steps taken, i.e. if incorrect, then longer negative traces should get larger negative reward.  Correct traces that are shorter should get more positive reward.
4. Fine-tune the model on these reasoning traces with mix of other data as usual.
5. Use this model to generate reasoning traces for slightly harder problems this new model can barely do.

Repeat generation of reasoning traces and fine-tuning until the model can do the hardest problems, such that the scope of reasoning traces as consumed more types of problems (but not all types since not always required).

## Speculations

1. MCTS, ToT, agents, etc. not required at training or inference time.
2. Human labeling or human verification of reasoning traces are not required.
3. Fine-tuned models for verification are not required, whichever step.
4. RLHF is not strictly required, just DPO.
5. OpenAI is using Deep RL for training the reasoning traces, but I don't think this is required.

## Project Goals

1. Generate reasoning traces using the proposed approach.
2. Fine-tune a model on the generated reasoning traces.
3. Evaluate the performance and compare it with existing models with zero-shot, few-shot, CoT, etc.

Other projects:
* Key difference with [Raspberry](https://github.com/daveshap/Raspberry) is that they are focused on hard prompts, while we think a progressive learning approach with repeated fine-tuning will bootstrap towards o1.
* Key difference with [g1](https://github.com/bklieger-groq/g1) is that they are focused on o1-like behavior alone, without emphasis how to fine-tune towards o1.
* Anthropic and Google API support of prompt caching means much cheaper to run.  vLLM supports prefix caching that helps that too.

## Current Status

This project is in its initial stages. Results and comparisons will be added as they become available.

TODO:
- [x] Setup basic anthropic case with prompt caching
- [x] Setup basic streamlit app to easily monitor outputs
- [x] Look for community support
- [x] Every (say) 9 steps, ask if model thinks it has final answer, and if so then ask it to place that answer in <final_answer> xml tags for extraction and termination of the reasoning trace.
- [x] Add backoff
- [x] Add ollama, google, azure, openai, groq, anthropic APIs with prompt caching for anthropic
- [x] Add high-level summary of blocks of text like o1
- [ ] Improve system prompt, vary it as well or separately from user next prompts
- [ ] Add verifier that samples window of history and separately critiques the assistant output
- [ ] Use existing datasets with ground truth to identify problems for which CoT achieves success after some trials
- [ ] Harvest CoT-friendly prompts and collect positive and negative reasoning traces
- [ ] Fine-tune with DPO including with mix of normal data as well with similar distribution 
- [ ] Repeat on next round of CoT-friendly prompts excluding original prompts, so can bootstrap
- [ ] Fine-tune on top of Fine-tune including with mix of normal data as well with similar distribution
- [ ] Repeat overall until bootstrap-repeat ones way to a smarter model

## Contributing

We welcome contributions from the community. Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to participate.

Issues:
- [ ] Continue button in app leaves grayed-out old chats, best if started cleanly
- [ ] Counting of tokens only shows up after hit continue, best if was every turn

## About the Author

Jonathan McKinney is the Director of Research at H2O.ai with a background in astrophysics and machine learning. His experience includes:

- Former Astrophysics Professor at UMD [B1][B2][B3][B4]
- 7 years of experience with AutoML products at H2O.ai [B5][B6]
- Recent work on fine-tuning LLMs, RAG, and AI Agents (h2oGPT) [B7][B8]
- See my other projects like [h2oGPT](https://github.com/h2oai/h2ogpt) and [prompt-engineering](https://github.com/pseudotensor/prompt-engineering)

## Disclaimer

This project is speculative and based on publicly available information about OpenAI's work. It is not affiliated with or endorsed by OpenAI.

## References

[1] https://openai.com/index/learning-to-reason-with-llms/

[B1] https://umdphysics.umd.edu/about-us/news/department-news/697-jon-mckinney-publishes-in-science-express.html

[B2] https://umdphysics.umd.edu/academics/courses/945-physics-420-principles-of-modern-physics.html

[B3] https://www.linkedin.com/in/jonathan-mckinney-32b0ab18/

[B4] https://scholar.google.com/citations?user=5L3LfOYAAAAJ&hl=en

[B5] https://h2o.ai/company/team/makers/

[B6] https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/

[B7] https://arxiv.org/abs/2306.08161

[B8] https://github.com/h2oai/h2ogpt

[P0] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models: https://arxiv.org/abs/2201.11903

[P1] STaR: Bootstrapping Reasoning With Reasoning: https://arxiv.org/abs/2203.14465

[P2] Let's Verify Step by Step: https://arxiv.org/abs/2305.20050

[P3] Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking: https://arxiv.org/abs/2403.09629

[P4] Think before you speak: Training Language Models With Pause Tokens: https://arxiv.org/abs/2310.02226

[P5] Nash Learning from Human Feedback: https://arxiv.org/abs/2312.00886

Related Projects:
* https://github.com/daveshap/Raspberry
* https://github.com/bklieger-groq/g1
* https://github.com/ranfysvalle02/llm-reasoning-illusion
* https://github.com/hijkzzz/Awesome-LLM-Strawberry

Related Videos:
* https://www.youtube.com/watch?v=tpun1uOKecc (Cascading prompts with repeated CoT)
