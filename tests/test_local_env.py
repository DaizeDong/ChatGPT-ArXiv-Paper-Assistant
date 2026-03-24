from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.utils import local_env


class TestLocalEnv(unittest.TestCase):
    def test_load_local_env_uses_dotenv_then_dotenv_local_without_overwriting_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            (repo_root / ".env").write_text(
                "OPENAI_API_KEY=dotenv-key\nX_BEARER_TOKEN=dotenv-token\n",
                encoding="utf-8",
            )
            (repo_root / ".env.local").write_text(
                "OPENAI_API_KEY=local-key\nX_BEARER_TOKEN='x-token'\nGITHUB_TOKEN=gh-local\n",
                encoding="utf-8",
            )
            with patch.object(local_env, "REPO_ROOT", repo_root), patch.dict(os.environ, {"OPENAI_API_KEY": "existing"}, clear=True):
                loaded = local_env.load_local_env()
                self.assertEqual(os.environ["OPENAI_API_KEY"], "existing")
                self.assertEqual(os.environ["X_BEARER_TOKEN"], "x-token")
                self.assertEqual(os.environ["GITHUB_TOKEN"], "gh-local")
                self.assertEqual(loaded["X_BEARER_TOKEN"], "x-token")
                self.assertNotIn("OPENAI_API_KEY", loaded)


if __name__ == "__main__":
    unittest.main()
