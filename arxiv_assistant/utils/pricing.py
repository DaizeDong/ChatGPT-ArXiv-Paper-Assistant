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

    # https://platform.openai.com/docs/pricing
    # The official ChatGPT API.
    # [LAST UPDATE: 2025.9.5]
    "gpt-5": {"prompt": 1.25, "completion": 10.00, "cache": 0.125},
    "gpt-5-mini": {"prompt": 0.25, "completion": 2.00, "cache": 0.025},
    "gpt-5-nano": {"prompt": 0.05, "completion": 0.40, "cache": 0.005},
    "gpt-5-chat-latest": {"prompt": 1.25, "completion": 10.00, "cache": 0.125},
    "gpt-4.1": {"prompt": 2.0, "completion": 8.0, "cache": 0.5},
    "gpt-4.1-mini": {"prompt": 0.4, "completion": 1.6, "cache": 0.1},
    "gpt-4.1-nano": {"prompt": 0.1, "completion": 0.4, "cache": 0.025},
    "gpt-4o": {"prompt": 2.5, "completion": 10.0, "cache": 1.25},
    "gpt-4o-2024-05-13": {"prompt": 5.0, "completion": 15.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6, "cache": 0.075},
    "gpt-realtime": {"prompt": 4.00, "completion": 16.00, "cache": 0.40},
    "gpt-4o-realtime-preview": {"prompt": 5.0, "completion": 20.0, "cache": 2.5},
    "gpt-4o-mini-realtime-preview": {"prompt": 0.6, "completion": 2.4, "cache": 0.3},
    "gpt-audio": {"prompt": 2.5, "completion": 10.0},
    "gpt-4o-audio-preview": {"prompt": 2.5, "completion": 10.0},
    "gpt-4o-mini-audio-preview": {"prompt": 0.15, "completion": 0.6},
    "o1": {"prompt": 15.0, "completion": 60.0, "cache": 7.5},
    "o1-pro": {"prompt": 150.0, "completion": 600.0},
    "o3": {"prompt": 10.0, "completion": 40.0, "cache": 2.5},
    "o3-pro": {"prompt": 20.0, "completion": 80.0},
    "o3-deep-research": {"prompt": 10.0, "completion": 40.0, "cache": 2.5},
    "o4-mini": {"prompt": 1.1, "completion": 4.4, "cache": 0.275},
    "o4-mini-deep-research": {"prompt": 2.0, "completion": 8.0, "cache": 0.5},
    "o3-mini": {"prompt": 1.1, "completion": 4.4, "cache": 0.55},
    "o1-mini": {"prompt": 1.1, "completion": 4.4, "cache": 0.55},
    "codex-mini-latest": {"prompt": 1.5, "completion": 6.0, "cache": 0.375},
    "gpt-4o-mini-search-preview": {"prompt": 0.15, "completion": 0.6},
    "gpt-4o-search-preview": {"prompt": 2.5, "completion": 10.0},
    "computer-use-preview": {"prompt": 3.0, "completion": 12.0},
    "gpt-image-1": {"prompt": 5.0, "completion": 1.25},

    # https://ai.google.dev/pricing
    # The official Gemini API.
    # Here are the prices for the "Pay-as-you-go" plan instead of the free plan.
    # [LAST UPDATE: 2025.9.5]
    "gemini-2.5-pro": {"prompt": 1.25, "completion": 10, "cache": 0.31},  # Prompts up to 200k tokens here. Prices for prompts longer than 200k are ~doubled.
    "gemini-2.5-flash": {"prompt": 0.3, "completion": 2.5, "cache": 0.075},
    "gemini-2.5-flash-lite": {"prompt": 0.1, "completion": 0.4, "cache": 0.025},
    "gemini-2.0-flash": {"prompt": 0.1, "completion": 0.4, "cache": 0.0025},
    "gemini-2.0-flash-lite": {"prompt": 0.075, "completion": 0.3},
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.3, "cache": 0.01875},  # Prompts up to 128k tokens here. Prices for prompts longer than 128k are doubled.
    "gemini-1.5-flash-8b": {"prompt": 0.0375, "completion": 0.15, "cache": 0.01},  # Prompts up to 128k tokens here. Prices for prompts longer than 128k are doubled.
    "gemini-1.5-pro": {"prompt": 1.25, "completion": 5, "cache": 0.3125},  # Prompts up to 128k tokens here. Prices for prompts longer than 128k are doubled.

    # https://api-docs.deepseek.com/quick_start/pricing
    # The official DeepSeek API.
    # [LAST UPDATE: 2025.9.5]
    "deepseek-chat": {"prompt": 0.56, "completion": 1.68, "cache": 0.07},
    "deepseek-reasoner": {"prompt": 0.56, "completion": 1.68, "cache": 0.07},
}
