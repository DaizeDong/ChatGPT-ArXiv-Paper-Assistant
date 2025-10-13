# Getting a Free OpenAI API key form GitHub Copilot Free Plan

You can get a **free** API Key to call GPT-4o with a [rate limit](https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits) from [GitHub](https://github.com/marketplace/models/azure-openai/gpt-4o). Its daily limit is enough for filtering ArXiv papers.

For reference, GPT-4o is on the `High` rate limit tier, which allows the following usage for Copilot Free plan:

| 	Rate limits         | Copilot Free      |
|----------------------|-------------------|
| 	Requests per minute | 10                |
| Requests per day     | 50                |
| Tokens per request   | 8000 in, 4000 out |
| Concurrent requests  | 2                 |

As a cost estimate:

- Filtering 267 papers by titles with `batch_size=40` takes 7 queries with an average of 1,798 prompt tokens and 144 completion tokens each.

- Filtering 123 papers by abstracts with `batch_size=12` takes 11 queries with an average of 4,477 prompt tokens and 739 completion tokens each.

This costs $0 under the [rate limit](https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits) of the Copilot Free plan.
