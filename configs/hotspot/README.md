# Hotspot Registries

This folder keeps the runtime JSON registries used by the hotspot pipeline.

- `official_blogs.json`: first-party vendor blogs and newsrooms, each with the fetch mode its site needs (`rss`, `sitemap`, `generic_html`, `playwright`)
- `analysis_feeds.json`: analyst and commentary feeds
- `model_hubs.json`: organisations whose releases are WEIGHTS rather than posts. Some labs never write an announcement, so the hub's model index is the release event; measured over a year of this archive, 34 of 93 frontier models being discussed had no announcement collected anywhere. Add an org only after checking `https://huggingface.co/api/models?author=<org>&sort=createdAt&direction=-1` actually returns rows for it.
- `roundup_sites.json`: roundup, newsletter, and editorial source registry
- `x_authority_seeds.json`: curated official/company/researcher seed accounts plus following-graph expansion settings used to build the dynamic X authority registry
- `x_authority_inventory.json`: tracked X authority inventory; this is the runtime source that the hotspot pipeline reads every day, and it is refreshed explicitly by the registry update script or workflow

If a hotspot-related file is not read by the pipeline at runtime, it should live in `docs/` instead of this folder.
