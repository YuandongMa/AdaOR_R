# AdaOR Project Changes

This package is a directly downloadable variant of the uploaded AdaOR project with a focused architectural update:

## Implemented change
- Added a **Router inside MSRM** in `src/model/feature_extractor_biovis.py`

## Router design
The MSRM module now contains an internal `HiddenStateRouter` that:
- splits hidden representations into:
  - texture hidden state
  - structure hidden state
  - spatial hidden state
- predicts `hp ∈ R^(N×1)` to filter spatially informative tokens
- predicts `gamma ∈ R^(3×1)` to adaptively weight the three hidden branches

## Forward compatibility
- Existing `BioVisionFeatureExtractor(x)` usage still works
- Optional router statistics can be returned with:
  - `BioVisionFeatureExtractor(...).forward(x, return_router_stats=True)`

## Scope note
This package updates the explicit AdaOR bio-inspired feature extractor file and README description.
It does **not** refactor every other encoder/decoder path in the project into the Router-based MSRM design.
