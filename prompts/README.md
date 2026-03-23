# Prompt Layout

The prompt files are organized by pipeline instead of keeping every prompt in one flat folder.

```text
prompts/
  paper/
    system_prompt.txt
    paper_topics.txt
    score_criteria.txt
    postfix_prompt_title.txt
    postfix_prompt_abstract.txt
    example_prompt_structure.md
    templates/
      paper_topics.template.txt
      score_criteria.template.txt
  hotspot/
    system_prompt.txt
    screening_criteria.txt
    postfix_prompt_screening.txt
    digest_writer.txt
  monthly/
    system_prompt.txt
    criteria.txt
    postfix_prompt.txt
```

Notes:

- `paper/` drives the daily arXiv filtering pipeline.
- `hotspot/` drives daily AI hotspot screening and digest synthesis.
- `monthly/` drives monthly summary regeneration.
- `paper/templates/` stores starter paper prompt files.
- `paper/example_prompt_structure.md` is a lightweight paper-prompt reference.

Code now loads prompts through a centralized resolver with backward-compatible fallback paths, so the runtime reads the organized layout first.
