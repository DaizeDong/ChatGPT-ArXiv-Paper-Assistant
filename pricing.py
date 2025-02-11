MODEL_PRICING = {
    # name: prompt, cache, completion

    # https://api.datapipe.app/pricing
    # This is a third-party ChatGPT API whose prices are much more expensive than the official ones.
    # Comment these lines when you are using the official ChatGPT API.
    # [LAST UPDATE: 2025.1.10]
    # "gpt-3.5-turbo": {"prompt": 7.5, "completion": 22.5},
    # "gpt-3.5-turbo-0125": {"prompt": 2.5, "completion": 7.5},
    # "gpt-4": {"prompt": 150, "completion": 300},
    # "gpt-4-32k": {"prompt": 300, "completion": 600},
    # "gpt-4-dalle": {"prompt": 300, "completion": 600},
    # "gpt-4-v": {"prompt": 300, "completion": 600},
    # "gpt-4-all": {"prompt": 300, "completion": 300},
    # "gpt-4-turbo": {"prompt": 300, "completion": 900},
    # "gpt-4-turbo-preview": {"prompt": 50, "completion": 150},
    # "gpt-4o": {"prompt": 25, "completion": 100},
    # "gpt-4o-2024-08-06": {"prompt": 25, "completion": 100},
    # "gpt-4o-2024-11-20": {"prompt": 25, "completion": 100},
    # "gpt-4o-all": {"prompt": 300, "completion": 1200},
    # "gpt-4o-mini": {"prompt": 7.5, "completion": 30},
    # "gpt-ask-internet": {"prompt": 50, "completion": 50},

    # https://openai.com/api/pricing
    # The official ChatGPT API.
    # [LAST UPDATE: 2025.1.28]
    "gpt-3.5-turbo": {"prompt": 7.5, "completion": 22.5},
    "gpt-3.5-turbo-0125": {"prompt": 0.5, "completion": 1.5},
    "gpt-3.5-turbo-instruct": {"prompt": 1.5, "completion": 2},
    "gpt-3.5-turbo-1106": {"prompt": 1, "completion": 2},
    "gpt-3.5-turbo-0613": {"prompt": 1.5, "completion": 2},
    "gpt-3.5-turbo-16k-0613": {"prompt": 3, "completion": 4},
    "gpt-3.5-turbo-0301": {"prompt": 1.5, "completion": 2},
    "gpt-4": {"prompt": 30, "completion": 60},
    "gpt-4-32k": {"prompt": 60, "completion": 120},
    "gpt-4-turbo": {"prompt": 10, "completion": 30},
    "gpt-4-turbo-2024-04-09": {"prompt": 10, "completion": 30},
    "gpt-4-0125-preview": {"prompt": 10, "completion": 30},
    "gpt-4-1106-preview": {"prompt": 10, "completion": 30},
    "gpt-4-vision-preview": {"prompt": 10, "completion": 30},
    "gpt-4o-latest": {"prompt": 5, "completion": 15},
    "gpt-4o": {"prompt": 2.5, "completion": 10, "cache": 1.25},
    "gpt-4o-2024-08-06": {"prompt": 2.5, "completion": 10, "cache": 1.25},
    "gpt-4o-2024-11-20": {"prompt": 2.5, "completion": 10, "cache": 1.25},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6, "cache": 0.075},
    "gpt-4o-mini-2024-07-18": {"prompt": 0.15, "completion": 0.6, "cache": 0.075},
    "o1": {"prompt": 15, "completion": 60, "cache": 7.5},
    "o1-2024-12-17": {"prompt": 15, "completion": 60, "cache": 7.5},
    "o1-preview": {"prompt": 15, "completion": 60, "cache": 7.5},
    "o1-preview-2024-09-12": {"prompt": 15, "completion": 60, "cache": 7.5},
    "o1-mini": {"prompt": 3, "completion": 12, "cache": 1.5},
    "o1-mini-2024-09-12": {"prompt": 3, "completion": 12, "cache": 1.5},

    # https://ai.google.dev/pricing
    # The official Gemini API.
    # Here are the prices for the "Pay-as-you-go" plan instead of the free plan.
    # [LAST UPDATE: 2025.2.10]
    "gemini-2.0-flash": {"prompt": 0.1, "completion": 0.4, "cache": 0.0025},
    "gemini-2.0-flash-lite-preview-02-05": {"prompt": 0.075, "completion": 0.3, "cache": 0.01875},
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.3, "cache": 0.01875}, # Prompts up to 128k tokens here. Prices for prompts longer than 128k are doubled.
    "gemini-1.5-flash-8b": {"prompt": 0.0375, "completion": 0.15, "cache": 0.01}, # Prompts up to 128k tokens here. Prices for prompts longer than 128k are doubled.
    "gemini-1.5-pro": {"prompt": 1.25, "completion": 5, "cache": 0.3125 }, # Prompts up to 128k tokens here. Prices for prompts longer than 128k are doubled.

    # https://api-docs.deepseek.com/quick_start/pricing
    # The official DeepSeek API.
    # [LAST UPDATE: 2025.1.28]
    "deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
}
