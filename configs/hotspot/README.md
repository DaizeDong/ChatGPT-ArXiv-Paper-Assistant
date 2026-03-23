# Hotspot Registries

This folder keeps the runtime JSON registries used by the hotspot pipeline.

- `roundup_sites.json`: roundup, newsletter, and editorial source registry
- `x_authority_seeds.json`: curated official/company/researcher seed accounts plus following-graph expansion settings used to build the dynamic X authority registry
- `x_authority_inventory.json`: tracked X authority inventory; this is the runtime source that the hotspot pipeline reads every day, and it is refreshed explicitly by the registry update script or workflow

If a hotspot-related file is not read by the pipeline at runtime, it should live in `docs/` instead of this folder.
