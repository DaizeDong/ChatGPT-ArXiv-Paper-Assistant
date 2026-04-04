import unittest

from arxiv_assistant.paper_topics import (
    build_daily_topic_bundle,
    build_hotspot_paper_bundle,
    ensure_topic_fields,
    get_topic_registry,
    group_sorted_papers_by_topic,
    sort_paper_mapping_for_daily_display,
)


class PaperTopicsTests(unittest.TestCase):
    def test_topic_registry_uses_expected_order(self):
        registry = get_topic_registry()
        self.assertEqual(
            list(registry.topic_ids),
            [
                "architecture_training",
                "efficiency_scaling",
                "representation_structure",
                "memory_systems",
                "world_models_open_ended_rl",
            ],
        )

    def test_ensure_topic_fields_normalizes_old_category_labels(self):
        entry = ensure_topic_fields(
            {
                "arxiv_id": "2501.00001",
                "title": "Distributed memory-efficient training",
                "PRIMARY_CATEGORY": "High Performance Computing",
            },
            arxiv_id="2501.00001",
        )
        self.assertEqual(entry["PRIMARY_TOPIC_ID"], "efficiency_scaling")
        self.assertEqual(entry["PRIMARY_TOPIC_LABEL"], "Efficiency, Compression, and Large-Scale Training")

    def test_group_sorted_papers_by_topic_preserves_sorted_projection(self):
        sorted_mapping = sort_paper_mapping_for_daily_display(
            {
                "2501.00001": {
                    "arxiv_id": "2501.00001",
                    "title": "Memory agent",
                    "authors": ["A"],
                    "abstract": "episodic memory for agents",
                    "SCORE": 18,
                    "RELEVANCE": 9,
                    "NOVELTY": 9,
                    "PRIMARY_TOPIC_ID": "memory_systems",
                },
                "2501.00002": {
                    "arxiv_id": "2501.00002",
                    "title": "World model",
                    "authors": ["B"],
                    "abstract": "world model planning",
                    "SCORE": 17,
                    "RELEVANCE": 9,
                    "NOVELTY": 8,
                    "PRIMARY_TOPIC_ID": "world_models_open_ended_rl",
                },
                "2501.00003": {
                    "arxiv_id": "2501.00003",
                    "title": "Another memory paper",
                    "authors": ["C"],
                    "abstract": "semantic memory",
                    "SCORE": 16,
                    "RELEVANCE": 9,
                    "NOVELTY": 7,
                    "PRIMARY_TOPIC_ID": "memory_systems",
                },
            }
        )
        grouped = group_sorted_papers_by_topic(list(sorted_mapping.values()))

        self.assertEqual([section["topic_id"] for section in grouped], ["memory_systems", "world_models_open_ended_rl"])
        self.assertEqual(
            [paper["arxiv_id"] for paper in grouped[0]["papers"]],
            ["2501.00001", "2501.00003"],
        )
        self.assertEqual(
            [paper["arxiv_id"] for paper in grouped[1]["papers"]],
            ["2501.00002"],
        )

    def test_build_daily_topic_bundle_contains_only_non_empty_sections(self):
        bundle = build_daily_topic_bundle(
            (2026, 3, 31),
            {
                "2501.00001": ensure_topic_fields(
                    {
                        "arxiv_id": "2501.00001",
                        "title": "Memory paper",
                        "authors": ["A"],
                        "abstract": "episodic memory",
                        "SCORE": 18,
                        "RELEVANCE": 9,
                        "NOVELTY": 9,
                        "PRIMARY_TOPIC_ID": "memory_systems",
                    },
                    arxiv_id="2501.00001",
                )
            },
        )
        self.assertEqual(bundle["meta"]["date"], "2026-03-31")
        self.assertEqual(len(bundle["topic_sections"]), 1)
        self.assertEqual(bundle["topic_sections"][0]["topic_id"], "memory_systems")

    def test_build_hotspot_paper_bundle_prioritizes_new_frontier_before_daily_hot(self):
        bundle = build_hotspot_paper_bundle(
            (2026, 4, 1),
            {
                "2604.00001": ensure_topic_fields(
                    {
                        "arxiv_id": "2604.00001",
                        "title": "Field-theoretic frontier model",
                        "authors": ["A"],
                        "abstract": "A new frontier architecture.",
                        "SCORE": 18,
                        "RELEVANCE": 9,
                        "NOVELTY": 9,
                        "PRIMARY_TOPIC_ID": "architecture_training",
                        "HOTSPOT_PAPER_TAGS": ["daily_hot", "new_frontier"],
                    },
                    arxiv_id="2604.00001",
                ),
                "2604.00002": ensure_topic_fields(
                    {
                        "arxiv_id": "2604.00002",
                        "title": "Daily hot systems paper",
                        "authors": ["B"],
                        "abstract": "A broadly important same-day paper.",
                        "SCORE": 17,
                        "RELEVANCE": 9,
                        "NOVELTY": 8,
                        "PRIMARY_TOPIC_ID": "efficiency_scaling",
                        "HOTSPOT_PAPER_TAGS": ["daily_hot"],
                    },
                    arxiv_id="2604.00002",
                ),
            },
            max_daily_hot=3,
            max_new_frontier=3,
        )

        self.assertEqual([section["kind"] for section in bundle["sections"]], ["new_frontier", "daily_hot"])
        self.assertEqual(bundle["sections"][0]["paper_ids"], ["2604.00001"])
        self.assertEqual(bundle["sections"][1]["paper_ids"], ["2604.00002"])


if __name__ == "__main__":
    unittest.main()
