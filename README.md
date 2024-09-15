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
