# ChatGPT ArXiv Paper Assistant: A Daily ArXiv Scanner

> *[Last update: 2025-9-5]*
> This is an enhanced version of the [GPT paper assistant](https://github.com/tatsu-lab/gpt_paper_assistant).
> I fixed all known bugs and added various new features to make it easier to use.
> See the [changelog](CHANGELOG.md) for details.

This repo implements a very simple daily scanner for Arxiv that uses OpenAI API to find papers you might find interesting.
It will run daily via github actions and can post this information to slack via a bot or just render it in a static github-pages website.
The results will be pushed to the `auto_update` branch automatically.

## Quickstart

This is the minimal necessary steps to get the scanner to run. It is highly recommended to read the whole thing to decide what you want to run.

### Running on github actions

1. Copy/fork this repo to a new github repo and [enable scheduled workflows](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow) if you fork it.
2. Copy `prompts/templates/paper_topics.template.txt` to `prompts/paper_topics.txt` and fill it out with the types of papers you want to follow.
3. Copy `config/templates/authors.template.txt` to `config/authors.txt` and list the authors you actually want to follow. The numbers behind the author are important. They are semantic scholar author IDs which you can find by looking up the authors on semantic scholar and taking the numbers at the end of the URL.
4. Set your desired ArXiv categories in `config/config.ini`.
5. Set your openai key `OPENAI_API_KEY` and base url `OPENAI_BASE_URL` (if you need one) as [github secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions#creating-secrets-for-a-repository). To can get a free one from GitHub, please reference [GUIDE_GITHUB_API.md](GUIDE_GITHUB_API.md).
6. In your repo settings, set github page build sources to be [github actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow).

At this point your bot should run daily and publish a static website. The results will be pushed to the `auto_update` branch automatically. You can test this by running the github action workflow manually.

**Optional**:

7. (Recommended) Adjust the content in `prompts/score_criteria.txt` by your requirements. For example, you can add some examples for each class for reference.
8. (Recommended) Take a look at `configs/config.ini` to tweak how things are filtered.
9. Get and set up a semantic scholar API key (`S2_KEY`) as a github secret. Otherwise the author search step will be very slow. (For now the keys are tight, so you may not be able to get one.)
10. [Set up a slack bot](https://api.slack.com/start/quickstart), get the OAuth key, set it to `SLACK_KEY` as a github secret.
11. Make a channel for the bot (and invite it to the channel), get its [Slack channel id](https://stackoverflow.com/questions/40940327/what-is-the-simplest-way-to-find-a-slack-team-id-and-a-channel-id), set it as `SLACK_CHANNEL_ID` in a github secret.
12. Set the github repo private to avoid github actions being [set to inactive after 60 days](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow).

Each day at 5am UTC, the bot will run and post to slack and publish a github pages website (see the `publish_md` and `cron_runs` actions for details).

### Running locally

The steps are generally the same as above, but you have to set up the environment via `requirements.txt`

Instead of passing credentials via github secrets, you have to set environment variables `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `SLACK_KEY`, `SLACK_CHANNEL_ID`.

To run everything, just call `main.py`

**Other notes:**

- You may also want to not push to slack, in which case set your desired output endpoint (json, markdown, slack) in the `dump_json`, `dump_md`, and `push_to_slack` fields of `config/config.ini`.
- If the semantic scholar API times out or is slow, you should get a [S2 api key](https://www.semanticscholar.org/product/api#api-key-form) and set it as `S2_KEY` in your environment variables.
  (due to the limitations of github actions, this will only help if the code is run locally)

**Making it run on its own:**

This whole thing takes almost no compute, so you can rent the cheapest VM from AWS, put this repo in it, install the `requirements.txt`
appropriately set up the environment variables and add the following crontab

```
0 5 * * * python ~/arxiv_scanner/main.py
```

This crontab will run the script every 5am UTC.

## Making the `paper_topics.txt` prompt

The `paper_topics.txt` file is used to generate the prompt for GPT. It is a list of topics that you want to follow.
One set of examples might be something like

```text
 1. New methodological improvements to RLHF or instruction-following which are specific fine-tuning steps that are taken to make language models better at following user instructions across a range of tasks.
    - Relevant: papers that discuss specific methods like RLHF, or instruction-tuning datasets, improving these methods, or analyzing them.
    - Not relevant: papers about adaptation to some task. Simply following instructions or inputs are not sufficient.
 2. Shows new powerful test set contamination or membership inference methods for language models. Test set contamination is the phenomenon where a language model observes a benchmark dataset during pretraining.
    - Relevant: test statistics that can detect contamination of benchmarks in language models. statistics that can provide guarantees are more interesting. membership inference methods that are general enough to apply to language models are also relevant.
    - Not relevant: any papers that do not consider language models, or that do not consider test set contamination.
 3. Shows a significant advance in the performance of diffusion language models.
    - Relevant: papers that study language models that are also diffusion models. Continuous diffusions are even more relevant, while discrete diffusions are less so.
    - Not relevant: papers about image diffusions like DALL-E or Stable Diffusion, or papers that do not explicitly mention language models or applications to text.
```

This is just a standard prompt, but being very specific can help, especially for things like 'diffusion language models' or 'instruction-following', where the LM can get confused about whether image diffusions are relevant, or if doing some task better is sufficient to improve instruction following.

You may also want to follow this with some general interest areas like

```text
In suggesting papers to your friend, remember that he enjoys papers on statistical machine learning, and generative modeling in natural language processing.
Your friend also likes learning about surprising empirical results in language models, as well as clever statistical tricks.
He does not want to read papers that are about primarily applications of methods to specific domains.
```

## Details of how it works

The script grabs a candidate set of ArXiv papers for a specific day, via the RSS feeds. To avoid double-announcing papers, it will only grab an RSS feed within the last day. To avoid missing papers, you'd want to run this every day.
**It filters out any `UPDATED` papers and announces only new ones, including the transferred (cross) ones from another topic.**

The filtering logic is pretty simple. We first check for author match.

1. Do a lookup of the authors on semantic scholar, getting a list of candidate matches.
2. Check the authors of the paper. If the author semantic scholar id matches someone in `authors.txt` it goes in the candidate set with a default score of `author_match_score`.

We then check for GPT-evaluated relevance. We do this in two steps.

1. Filter out any papers that have no authors with h-index above `h_cutoff` in `config.ini`. This is to reduce costs.
2. All remaining examples get batched, and are evaluated by a GPT model specified by `model` in `config.ini`. This step uses the [prompt](prompts/example.md) defined in `./prompts/`.
3. GPT scores the papers for relevance (to the topics in `config/papers_topics.txt`) and novelty (scale 1-10)
4. Papers are filtered if they have scores below either the relevance and novelty cutoffs in `config.ini`
5. Papers are given an overall score based on equal weight to relevance and novelty.

Finally, all papers are sorted by the max of their `author_match_score` and the sum of the GPT-rated relevance and novelty scores (the relevance and novelty scores will only show up in the final output if they are above the cutoff thresholds you set in the config file). Then the papers are rendered and pushed into their endpoints (text files or Slack).

## Acknowledgement

This repo and code was originally built by Tatsunori Hashimoto is licensed under the Apache 2.0 license.
Thanks to Chenglei Si for testing and benchmarking the GPT filter.
