import dataclasses
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class Paper:
    # paper class should track the list of authors, paper title, abstract, arxiv id
    authors: List[str]
    title: str
    abstract: str
    arxiv_id: str

    # add a hash function using arxiv_id
    def __hash__(self):
        return hash(self.arxiv_id)


def is_earlier(ts1, ts2):
    # compares two arxiv ids, returns true if ts1 is older than ts2
    return int(ts1.replace(".", "")) < int(ts2.replace(".", ""))


def batched(items, batch_size):
    # takes a list and returns a list of list with batch_size
    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]


def normalize_whitespace(string):
    """Replace multiple whitespaces with a single space."""
    return re.sub(r'\s+', ' ', string).strip()


def align_markdown_table(table_string: str, alignments: Union[Optional[str], List[Optional[str]], Tuple[Optional[str]]] = None) -> str:
    """
    Set the alignment of a markdown table.
    :param table_string: The markdown table string to align.
    :param alignments: The alignment directions. Can be "left", "center", or "right". (None denotes no change)
                       If `alignments` is a list, it should be the same length as the number of columns in the table.
    :return: The aligned markdown table string.
    """
    lines = table_string.split("\n")
    format_line = lines[1]  # "|:-----:|-----|:-----|-----:|"
    format_contents = format_line.split("|")[1:-1]  # [":-----:", "-----", ":-----", "-----:"]
    num_columns = format_line.count("|") - 1

    if not isinstance(alignments, (tuple, list)):
        alignments = [alignments] * num_columns

    new_format_contents = []
    for i in range(num_columns):
        if alignments[i] is None:
            new_format_contents.append(format_contents[i])
        elif alignments[i] == "left":
            new_format_contents.append(":" + "-" * len(format_contents[i]))
        elif alignments[i] in ("center", "centre"):
            new_format_contents.append(":" + "-" * len(format_contents[i]) + ":")
        elif alignments[i] == "right":
            new_format_contents.append("-" * len(format_contents[i]) + ":")
        else:
            raise ValueError(f"Invalid alignment: {alignments[i]}")
    new_format_line = "|".join([""] + new_format_contents + [""])
    new_lines = [lines[0], new_format_line] + lines[2:]
    new_table_string = "\n".join(new_lines)

    return new_table_string
