import configparser
import os
import time

from utils import create_dir


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


# now time
NOW_YEAR = time.strftime("%Y")
NOW_MONTH = time.strftime("%m")
NOW_DAY = time.strftime("%d")

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

# load config.ini
CONFIG = configparser.ConfigParser()
CONFIG.read("configs/config.ini")

# load authors.txt
with open("configs/authors.txt", "r", encoding="utf-8") as fopen:
    author_names, author_ids = parse_authors(fopen.readlines())
AUTHOR_ID_SET = set(author_ids)

# load prompts
with open("prompts/base_prompt.txt", "r", encoding="utf-8") as f:
    BASE_PROMPT = f.read()
with open("prompts/paper_topics.txt", "r", encoding="utf-8") as f:
    TOPIC_PROMPT = f.read()
with open("prompts/score_criteria.txt", "r", encoding="utf-8") as f:
    SCORE_PROMPT = f.read()
with open("prompts/postfix_prompt.txt", "r", encoding="utf-8") as f:
    POSTFIX_PROMPT = f.read()

# output path
OUTPUT_DEBUG_DIR = os.path.join(CONFIG["OUTPUT"]["output_path"], "debug", f"{NOW_YEAR}-{NOW_MONTH}")
OUTPUT_DEBUG_FILE_FORMAT = os.path.join(OUTPUT_DEBUG_DIR, f"{NOW_YEAR}-{NOW_MONTH}-{NOW_DAY}-" + "{}")
create_dir(OUTPUT_DEBUG_DIR)

OUTPUT_MD_DIR = os.path.join(CONFIG["OUTPUT"]["output_path"], "md", f"{NOW_YEAR}-{NOW_MONTH}")
OUTPUT_MD_FILE_FORMAT = os.path.join(OUTPUT_MD_DIR, f"{NOW_YEAR}-{NOW_MONTH}-{NOW_DAY}-" + "{}")
create_dir(OUTPUT_MD_DIR)

OUTPUT_JSON_DIR = os.path.join(CONFIG["OUTPUT"]["output_path"], "json", f"{NOW_YEAR}-{NOW_MONTH}")
OUTPUT_JSON_FILE_FORMAT = os.path.join(OUTPUT_JSON_DIR, f"{NOW_YEAR}-{NOW_MONTH}-{NOW_DAY}-" + "{}")
create_dir(OUTPUT_JSON_DIR)

print(f"OUTPUT_DEBUG_DIR: {OUTPUT_DEBUG_DIR}")
print(f"OUTPUT_DEBUG_FILE_FORMAT: {OUTPUT_DEBUG_FILE_FORMAT}")
print(f"OUTPUT_MD_DIR: {OUTPUT_MD_DIR}")
print(f"OUTPUT_MD_FILE_FORMAT: {OUTPUT_MD_FILE_FORMAT}")
print(f"OUTPUT_JSON_DIR: {OUTPUT_JSON_DIR}")
print(f"OUTPUT_JSON_FILE_FORMAT: {OUTPUT_JSON_FILE_FORMAT}")
print(f"###################################################################")
