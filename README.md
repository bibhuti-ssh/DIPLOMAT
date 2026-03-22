# DIPLOMAT

This repository combines the training pipeline and dataset assets used for DIPLOMAT workplace negotiation modeling.

It is organized into two core components:

- **DSA_DPO/** contains the end-to-end DSA-DPO workflow for generating sessions, scoring outcomes, creating preference pairs, and preparing training-ready data.
- **PROWESS/** contains the dataset resources, including the first **600 sampled conversations** from the full PROWESS corpus.

## Components

- **DSA_DPO/**: Pipeline code and configs for data generation, DSA-DPO processing, and model training preparation.
- **PROWESS/**: Curated workplace negotiation dataset subset (first 600 samples).

## Notes

- Use the DSA_DPO module when you want to run the full training-data pipeline.
- Use the PROWESS module when you need the dataset subset for experimentation or evaluation.
- Detailed usage and implementation instructions are available in each subdirectory README.