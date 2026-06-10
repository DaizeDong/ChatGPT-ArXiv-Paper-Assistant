import time
from typing import Dict, List, Optional

import requests
from requests import Session
from retry import retry
from tqdm import tqdm


def get_paper_citations(
    arxiv_ids: List[str],
    *,
    api_key: Optional[str] = None,
    timeout: int = 15,
) -> Dict[str, dict]:
    """Batch-fetch citation counts for arXiv papers from the Semantic Scholar Graph API.

    Free, no key required (an ``api_key`` only lifts the rate limit). Issues ONE
    ``paper/batch`` request for the whole list (avoids per-paper 429s).

    Returns ``{arxiv_id: {"citationCount": int, "influentialCitationCount": int,
    "year": int|None}}``; papers S2 does not know are omitted. Degrade-safe: ANY
    failure (non-200 / 429 / network / parse) returns ``{}`` and never raises, so a
    caller's ranking is unchanged when S2 is unavailable.
    """
    ids = [a for a in (str(x).strip() for x in (arxiv_ids or [])) if a]
    if not ids:
        return {}
    headers = {"X-API-KEY": api_key} if api_key else {}
    try:
        response = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": "citationCount,influentialCitationCount,year,publicationDate"},
            headers=headers,
            json={"ids": [f"ARXIV:{a}" for a in ids]},
            timeout=timeout,
        )
        if response.status_code != 200:
            print(f"Warning: Semantic Scholar paper/batch returned {response.status_code}; skipping citation signal.")
            return {}
        data = response.json()
    except Exception as ex:  # network/timeout/JSON — degrade to no signal, never raise
        print(f"Warning: Semantic Scholar citation lookup failed ({ex}); skipping signal.")
        return {}
    if not isinstance(data, list):
        return {}
    out: Dict[str, dict] = {}
    for arxiv_id, entry in zip(ids, data):
        if not isinstance(entry, dict):
            continue  # S2 returns null for unknown ids
        out[arxiv_id] = {
            "citationCount": int(entry.get("citationCount") or 0),
            "influentialCitationCount": int(entry.get("influentialCitationCount") or 0),
            "year": entry.get("year"),
        }
    return out


def get_author_batch(
    session: Session,
    ids: List[str],
    S2_API_KEY: str,
    fields: str = "name,hIndex,citationCount",
    **kwargs,
) -> List[Dict]:
    # TODO: seems not used in the codebase. remove if not needed
    # gets a batch of authors. analogous to author batch
    params = {
        "fields": fields,
        **kwargs,
    }
    if S2_API_KEY is None:
        headers = {}
    else:
        headers = {
            "X-API-KEY": S2_API_KEY,
        }
    body = {
        "ids": ids,
    }

    with session.post(
        "https://api.semanticscholar.org/graph/v1/author/batch",
        params=params,
        headers=headers,
        json=body,
    ) as response:
        response.raise_for_status()
        return response.json()


@retry(tries=10, delay=5.0)
def get_one_author(session, author: str, S2_API_KEY: str) -> str:
    # query the right endpoint https://api.semanticscholar.org/graph/v1/author/search?query=adam+smith
    params = {"query": author, "fields": "authorId,name,hIndex", "limit": "10"}
    if S2_API_KEY is None:
        headers = {}
    else:
        headers = {"X-API-KEY": S2_API_KEY}
    with session.get(
        "https://api.semanticscholar.org/graph/v1/author/search",
        params=params,
        headers=headers,
    ) as response:
        response.raise_for_status()
        response_json = response.json()
        if len(response_json["data"]) >= 1:
            return response_json["data"]
        else:
            return None


def get_authors(
    all_authors: List[str], S2_API_KEY: str, config: Optional[Dict], **kwargs
):
    # first get the list of all author ids by querying by author names
    author_metadata_dict = {}
    with Session() as session:
        for author in tqdm(all_authors):
            try:
                auth_map = get_one_author(session, author, S2_API_KEY)
            except Exception as ex:
                # if config["OUTPUT"].getboolean("debug_messages"):
                print(f"Exception happened: Failed to get author info ({ex.args})")
                auth_map = None
            if auth_map is not None:
                author_metadata_dict[author] = auth_map
            # add a 20ms wait time to avoid rate limiting
            # otherwise, semantic scholar aggressively rate limits, so do 1.0s
            if S2_API_KEY is not None:
                time.sleep(0.02)
            else:
                time.sleep(1.0)
    return author_metadata_dict
