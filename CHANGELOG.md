# Changelog

### 2025-9-5

- Added GPT-5 and updated prices.
- Updated guide for Github Copilot Free plan

### 2025-5-27

- Added system prompts for GPT filtering.

### 2025-4-3

- Added retries for ArXiv API calls.
- Rearranged date formats from `MM/DD/YYYY` to `YYYY-MM-DD`.

### 2025-3-25

- Supported the identification of title filtering prompts.
- Moved changelogs and prompt examples out of the readme file.

### 2025-3-20

- Supported getting papers on any specified date through arXiv API.
- Enhanced the logic of getting papers from RSS feeds.
- Added a script for remedying papers for missed dates.

### 2025-3-13

- Rearranged the file structure and cleaned some unused code snippets.

### 2025-2-19

- Added retrying for failed completion calls.
- Fixed the output file name, which will first follow ArXiv update time instead of local time.

### 2025-2-18

- Fixed a paper formatting bug which destroyed the performance of title filtering.
- Added retrying logic for GPT filtering so that there will be no paper missed.
- Added toggles that control the title/abstract filtering.
- Enhanced the debugging information by recording more logs and dumping more debug files.

### 2025-2-11

- Added a rate limit to API calls.

### 2025-2-3

- Fixed a bug that mistakenly filters all papers with high h-index.

### 2025-1-31

- Updated all github actions to the latest version.

### 2025-1-29

- Supported price calculation for cache tokens.
- Updated the price for `deepseek-chat` and `deepseek-reasoner`.

### 2025-1-28

- Fixed adaptive batch size when `paper_num <= adaptive_threshold`.
- Fixed the rename when `output.md` already exists.
- Added details in the return information for selected/filtered papers.

### 2025-1-25

- Fixed the exception when no paper is available.

### 2025-1-22

- Added a function that adaptively scales the `batch_size` by the number of papers.
- Supported detailed logging the cost of prompt and completion tokens.
- Adjusted the format of prompts to better utilize ChatGPT cache.

### 2025-1-21

- Fixed the auto-push workflow.
- Supported setting prompts for scoring.

### 2025-1-18

- Fixed the invalid retry logic for author searching.

### 2025-1-17

- Added a workflow that automatically pushes outputs to the `auto_update` branch.
- Added a toggle that decides whether to search authors before paper filtering.
- Rearranged the output directory, separating the formal outputs and debug logs.
- Enhanced the logging logic. Now it prints out more information about preserved papers and costs.

### 2025-1-10

- Set the version of `httpx` package to `0.27.2` for compatibility.
- Supported setting the `base_url` for OpenAI API.
- Supported counting costs for the latest GPT-4o series models.

### 2024-2-15

- Fixed a bug with author parsing in the RSS format.
- Cost estimates for title filtering being off.
- Crash when 0 papers are on the feed.

### 2024-2-7

- Fixed a critical issue from ArXiv changing their RSS format.
- Added and enabled a title filtering to reduce costs.
