# open-strawberry

An open-source implementation inspired by OpenAI's Strawberry algorithm.

## Background

open-strawberry is based on speculations about OpenAI's Strawberry, a refined search-generation algorithm for generating and verifying training data. This project aims to recreate a similar system using open-source tools and methodologies.

### Key Concepts

- **Q***: A hypothetical primordial search-generation algorithm developed by OpenAI to generate training data.
- **Strawberry**: An advanced search-generation algorithm by OpenAI for generating and verifying training data.
- **Orion**: A class of models fine-tuned on Strawberry data, including o1-mini, o1-preview, o1, o1-ioi, etc. [1]

## Proposed Methodology

1. Bootstrap using SFT-instruction tuned models and their chat history.
2. Implement a prompt system that guides the LLM to take incremental steps towards a solution.
3. Generate multi-turn chat reasoning traces, periodically checking for a final answer.
4. Employ a verification system to check for errors in the chat history.
5. Generate multiple reasoning traces per problem.
6. Apply this process to a large set of problems with verifiable ground truths.
7. Select correct reasoning traces for each problem.
8. Fine-tune a model using the selected reasoning traces.

## Speculations

1. MCTS, ToT, agents, etc. not required at training or inference time.
2. Bootstrapping is key.
   * Identify problems the instruct model can do barely with strong CoT and high temperature for some number of fixed (e.g. 20) repeats.
   * Fine-tune the model on these reasoning traces with mix of other data as usual.
   * Use this model to generate reasoning traces for slightly harder problems this new model can barely do.
   * Repeat until the model can do the hardest problems, and the scope of reasoning traces as consumed more types of problems (but not all types since not always required).
3. Emphasize first principles thinking.

## Project Goals

1. Generate reasoning traces using the proposed approach.
2. Fine-tune a model on the generated reasoning traces.
3. Evaluate the performance and compare it with existing models.

## Current Status

This project is in its initial planning stages. Results and comparisons will be added as they become available.

## Contributing

We welcome contributions from the community. Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to participate.

## About the Author

Jonathan McKinney is the Director of Research at H2O.ai with a background in astrophysics and machine learning. His experience includes:

- Former Astrophysics Professor at UMD [B1][B2][B3][B4]
- 7 years of experience with AutoML products at H2O.ai [B5][B6]
- Recent work on fine-tuning LLMs, RAG, and AI Agents (h2oGPT) [B7][B8]

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
