"""Importing the paper pipeline must not require an OpenAI key any more."""
import configparser
import unittest

from arxiv_assistant.utils.llm_gateway import legacy_openai_key_missing


def _config(backend=None):
    parser = configparser.ConfigParser()
    if backend is not None:
        parser.read_dict({"LLM": {"backend": backend}})
    return parser


class LegacyKeyGateTests(unittest.TestCase):
    def test_the_whole_truth_table(self):
        cases = [
            # (name, api_key, backend, must_raise)
            ("auto + no key", None, "auto", False),
            ("auto + empty key", "", "auto", False),
            ("auto + real key", "sk-x", "auto", False),
            ("llmcall + no key", None, "llmcall", False),
            ("agent + no key", None, "agent", False),
            ("openai + real key", "sk-x", "openai", False),
            ("openai + no key", None, "openai", True),
            ("openai + empty key", "", "openai", True),
            ("no [LLM] section + no key", None, None, False),
            ("openai with padding/case", None, "  OpenAI  ", True),
        ]
        for name, api_key, backend, must_raise in cases:
            with self.subTest(case=name):
                self.assertEqual(
                    legacy_openai_key_missing(api_key, _config(backend)),
                    must_raise,
                    name,
                )

    def test_empty_string_is_treated_as_missing(self):
        """An unset CI secret expands to "" , not to None.

        Accepting it as a key defers the failure to a 401 on the first batch,
        which reads as a model failure rather than a configuration one. That
        misattribution is the specific reason the last outage went unnoticed.
        """
        self.assertTrue(legacy_openai_key_missing("", _config("openai")))
        # Whitespace-only is not a key either. A secret that expanded to spaces
        # is just as unusable as one that expanded to nothing.
        self.assertTrue(legacy_openai_key_missing("   ", _config("openai")))
        self.assertTrue(legacy_openai_key_missing("\n\t", _config("openai")))

    def test_a_broken_config_object_does_not_crash_the_gate(self):
        class _Hostile:
            def has_section(self, name):
                raise RuntimeError("boom")

        # Degrades to "auto", which never demands a key.
        self.assertFalse(legacy_openai_key_missing(None, _Hostile()))

    def test_main_pipeline_imports_without_the_legacy_key(self):
        """The regression that matters: main.py's import chain must not need a key.

        Asserted by source inspection rather than by importing environment, whose
        import performs network IO. The unconditional raise must be gone and the
        guarded one must be in place.
        """
        import pathlib

        import arxiv_assistant.environment as env_module

        source = pathlib.Path(env_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn(
            'raise ValueError("OpenAI key is not set', source,
            "the unconditional import-time key requirement is back",
        )
        self.assertIn("legacy_openai_key_missing(OPENAI_API_KEY, CONFIG)", source)


if __name__ == "__main__":
    unittest.main()
