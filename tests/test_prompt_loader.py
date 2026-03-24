from __future__ import annotations

import os
import unittest
from pathlib import Path

from arxiv_assistant.utils.prompt_loader import resolve_prompt_path


class TestPromptLoader(unittest.TestCase):
    def test_resolve_prompt_paths_use_reorganized_layout(self) -> None:
        self.assertEqual(resolve_prompt_path("paper.topics").name, "paper_topics.txt")
        self.assertIn(f"prompts{os.sep}paper{os.sep}", str(resolve_prompt_path("paper.topics")))
        self.assertIn(f"prompts{os.sep}hotspot{os.sep}", str(resolve_prompt_path("hotspot.system_prompt")))
        self.assertIn(f"prompts{os.sep}monthly{os.sep}", str(resolve_prompt_path("monthly.system_prompt")))


if __name__ == "__main__":
    unittest.main()
