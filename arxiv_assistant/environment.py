import configparser
import os
from datetime import UTC, datetime

import feedparser

from arxiv_assistant.utils.io import create_dir


def parse_authors(lines):
    # parse the comma-separated author list, ignoring lines that are empty and starting with #
    author_ids = []
    authors = []
    for line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            continue
        author_split = line.split(",")
        author_ids.append(author_split[1].strip())
        authors.append(author_split[0].strip())
    return authors, author_ids


# load config.ini
CONFIG = configparser.ConfigParser()
CONFIG.read("configs/config.ini")

print({section: dict(CONFIG[section]) for section in CONFIG.sections()})
print(f"###################################################################")

# load authors.txt
with open("configs/authors.txt", "r", encoding="utf-8") as fopen:
    author_names, author_ids = parse_authors(fopen.readlines())
AUTHOR_ID_SET = set(author_ids)

# load prompts
with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
with open("prompts/paper_topics.txt", "r", encoding="utf-8") as f:
    TOPIC_PROMPT = f.read()
with open("prompts/score_criteria.txt", "r", encoding="utf-8") as f:
    SCORE_PROMPT = f.read()
with open("prompts/postfix_prompt_title.txt", "r", encoding="utf-8") as f:
    POSTFIX_PROMPT_TITLE = f.read()
with open("prompts/postfix_prompt_abstract.txt", "r", encoding="utf-8") as f:
    POSTFIX_PROMPT_ABSTRACT = f.read()

# keys
S2_API_KEY = os.environ.get("S2_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
SLACK_KEY = os.environ.get("SLACK_KEY")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")

print(f"S2_API_KEY: {S2_API_KEY}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
print(f"OPENAI_BASE_URL: {OPENAI_BASE_URL}")
print(f"SLACK_KEY: {SLACK_KEY}")
print(f"SLACK_CHANNEL_ID: {SLACK_CHANNEL_ID}")

if OPENAI_API_KEY is None:
    raise ValueError("OpenAI key is not set - please set OPENAI_API_KEY to your OpenAI key")

# now time
try:
    # get from ArXiv
    feed = feedparser.parse("https://export.arxiv.org/rss/cs.LG")  # use the cs.LG area
    if len(feed.entries) > 0:
        # Example `feed.published`: "Tue, 18 Feb 2025 00:00:00 -0500"
        parsed_time = datetime.strptime(feed.entries[0].published, "%a, %d %b %Y %H:%M:%S %z")
        NOW_TIME = parsed_time
        NOW_YEAR = int(NOW_TIME.strftime("%Y"))
        NOW_MONTH = int(NOW_TIME.strftime("%m"))
        NOW_DAY = int(NOW_TIME.strftime("%d"))
    else:
        raise ValueError("Feed does not contain any entries")
except Exception as ex:
    # use local time
    NOW_TIME = datetime.now(UTC)
    NOW_YEAR = int(NOW_TIME.strftime("%Y"))
    NOW_MONTH = int(NOW_TIME.strftime("%m"))
    NOW_DAY = int(NOW_TIME.strftime("%d"))

print(f"NOW_YEAR: {NOW_YEAR}")
print(f"NOW_MONTH: {NOW_MONTH}")
print(f"NOW_DAY: {NOW_DAY}")

# output path
OUTPUT_DEBUG_DIR = os.path.join(CONFIG["OUTPUT"]["output_path"], "debug", f"{NOW_YEAR}-{format(NOW_MONTH, '02d')}", f"{NOW_YEAR}-{format(NOW_MONTH, '02d')}-{format(NOW_DAY, '02d')}")
OUTPUT_DEBUG_FILE_FORMAT = os.path.join(OUTPUT_DEBUG_DIR, "{}")
create_dir(OUTPUT_DEBUG_DIR)

OUTPUT_MD_DIR = os.path.join(CONFIG["OUTPUT"]["output_path"], "md", f"{NOW_YEAR}-{format(NOW_MONTH, '02d')}")
OUTPUT_MD_FILE_FORMAT = os.path.join(OUTPUT_MD_DIR, f"{NOW_YEAR}-{format(NOW_MONTH, '02d')}-{format(NOW_DAY, '02d')}-" + "{}")
create_dir(OUTPUT_MD_DIR)

OUTPUT_JSON_DIR = os.path.join(CONFIG["OUTPUT"]["output_path"], "json", f"{NOW_YEAR}-{format(NOW_MONTH, '02d')}")
OUTPUT_JSON_FILE_FORMAT = os.path.join(OUTPUT_JSON_DIR, f"{NOW_YEAR}-{format(NOW_MONTH, '02d')}-{format(NOW_DAY, '02d')}-" + "{}")
create_dir(OUTPUT_JSON_DIR)

print(f"OUTPUT_DEBUG_DIR: {OUTPUT_DEBUG_DIR}")
print(f"OUTPUT_DEBUG_FILE_FORMAT: {OUTPUT_DEBUG_FILE_FORMAT}")
print(f"OUTPUT_MD_DIR: {OUTPUT_MD_DIR}")
print(f"OUTPUT_MD_FILE_FORMAT: {OUTPUT_MD_FILE_FORMAT}")
print(f"OUTPUT_JSON_DIR: {OUTPUT_JSON_DIR}")
print(f"OUTPUT_JSON_FILE_FORMAT: {OUTPUT_JSON_FILE_FORMAT}")
print(f"###################################################################")
