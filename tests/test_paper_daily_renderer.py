import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from arxiv_assistant.renderers.paper.render_daily import render_daily_md


class PaperDailyRendererTests(unittest.TestCase):
    def test_render_daily_md_groups_by_topic_and_hides_empty_sections(self):
        rendered = render_daily_md(
            all_entries=[],
            arxiv_paper_dict={},
            selected_paper_dict={
                "2501.00001": {
                    "arxiv_id": "2501.00001",
                    "title": "Memory systems for agents",
                    "authors": ["A"],
                    "abstract": "episodic memory for agents",
                    "COMMENT": "New memory consolidation mechanism.",
                    "RELEVANCE": 9,
                    "NOVELTY": 9,
                    "SCORE": 18,
                    "PRIMARY_TOPIC_ID": "memory_systems",
                },
                "2501.00002": {
                    "arxiv_id": "2501.00002",
                    "title": "World models for exploration",
                    "authors": ["B"],
                    "abstract": "world model and open-ended exploration",
                    "COMMENT": "World-model pretraining for transferable RL.",
                    "RELEVANCE": 9,
                    "NOVELTY": 8,
                    "SCORE": 17,
                    "PRIMARY_TOPIC_ID": "world_models_open_ended_rl",
                },
            },
            now_date=(2026, 3, 31),
            prompts=None,
            head_table=None,
        )

        self.assertIn("Topic Coverage", rendered)
        self.assertIn("Memory Structures and Agent Memory Systems (1)", rendered)
        self.assertIn("World Models, Exploration, and Open-Ended Reinforcement Learning (1)", rendered)
        self.assertNotIn("Architecture and Training Dynamics (0)", rendered)
        self.assertIn("#paper-2501-00001", rendered)
        self.assertIn('id="paper-2501-00002"', rendered)


if __name__ == "__main__":
    unittest.main()
