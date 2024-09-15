# OpenStrawberry
Open source version of OpenAI Strawberry

## Assumption

* Q* refers to a primordial search-generation algorithm to *generate training data* developed by OpenAI
* Strawberry is a refined search-generation algorithm to *generate and verify training data* developed by OpenAI
* Orion is a class of models that are fine-tuned on Strawberry data.  The first group is o1-mini, o1-preview, o1, o1-ioi, etc. [[1]](https://openai.com/index/learning-to-reason-with-llms/)

## Idea Background

Planning for OpenAI to release "Strawberry," I considered what they may be doing back around Sept 1, 2024.

1) Bootstrap using standard SFT-instruction tuned models using their chat history.
2) Using a prompt or system prompt, ask the LLM to take the most miniscule step toward the solution.
3) In the first turn, give the normal query alone.  Let the LLM follow the original system instruction.
4) Generate long reasoning traces in a multi-turn chat, every so often asking of the LLM should stop if it had a final answer.
5) Use a separate verifier system prompt to check a window of the chat history (say every 5 turns) for errors and insert that as a "But ..." in the chat history.
6) Generate (say) 10-100 such reasoning traces per problem.
7) Perform this for a large number of problems that have some ground truth (math, physics, coding, etc.)
8) Select reasoning traces that (for the given problem) were correct.
9) Use reasoning traces (chat history) to fine-tune a model.

Once o1 was released, I was surprised how fragmented the "Show chain of thought" was, so I assume they are doing something different, or possibly obfuscating the output.

## Plan

* Generate reasoning traces using above approach
* Fine-tune a model on reasoning traces

## Results

TBD



### My Background

* Prior Astrophysics Professor at UMD [[B1]](https://umdphysics.umd.edu/about-us/news/department-news/697-jon-mckinney-publishes-in-science-express.html) [[B2]](https://umdphysics.umd.edu/academics/courses/945-physics-420-principles-of-modern-physics.html) [[B3]](https://www.linkedin.com/in/jonathan-mckinney-32b0ab18/) [[B4]](https://scholar.google.com/citations?user=5L3LfOYAAAAJ&hl=en)
* Current Director of Research at H2O.ai [[B5]](https://h2o.ai/company/team/makers/)
* AutoML products at H2O.ai for last 7 years [[B6]](https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/)
* Fine-tuning LLMs [[B7]](https://arxiv.org/abs/2306.08161), RAG, and Agents for last 2 years (h2oGPT [[B8]](https://github.com/h2oai/h2ogpt))
* I have only one publication on LLMs, but I have lots of enthusiasm and some experience in the field.
* My eduction is in physics, not computer science, so I may be less adept at the details of LLMs.