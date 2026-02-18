Structural Compatibility Modeling for Apartment-Level Tenant Allocation in Co-Living

This repository contains the code, documentation, and modelling framework for a Master’s thesis on structural compatibility modeling in co-living environments.

Rather than predicting subjective compatibility directly, the project reframes tenant alignment as a latent-space structural problem, where individual behavioural and psychological representations are aggregated to evaluate apartment-level fit and support allocation decisions.

📘 Project Overview

Commercial co-living environments face recurring challenges related to tenant compatibility and shared living harmony.
Allocation decisions are often made with limited structured insight into interpersonal dynamics.

This thesis develops a two-stage compatibility framework:

Stage 1 – Personality Representation Learning

Large-scale personality data (IPIP / Big Five)

Dimensionality reduction (PCA)

Clustering to derive latent tenant representations

Stage 2 – Apartment-Level Structural Modeling

Integration of lifestyle and behavioural survey data

Construction of tenant embeddings

Aggregation to apartment-level mean vectors

Dispersion metrics within flats

Similarity-based compatibility scoring

Configurable demographic allocation constraints

Prototype decision-support interface

The goal is not to build a black-box predictor, but a transparent, configurable allocation support framework.

🧠 Conceptual Framing

Compatibility is operationalized as:

Distance in latent embedding space

Internal apartment dispersion

Structural alignment between tenant vector and apartment centroid

Rule-based demographic constraints (e.g., age bounds, gender balance)

The framework emphasizes:

Interpretability

Structural modeling over prediction

Ethical awareness in demographic controls

Operational feasibility for co-living operators

📊 Data Sources
Secondary Data

OpenPsychometrics Big Five dataset

Used for large-scale personality structure learning

Primary Data

Co-living survey (personality + lifestyle + behavioural attributes)

Limited response size

Supplemented via synthetic data generation for structural testing

No personally identifiable information is included in this repository.

🗂 Repository Structure
CoLiving-Thesis/
│
├── thesis/        # LaTeX source files for dissertation
├── notebooks/     # Stage 1 and Stage 2 Jupyter notebooks
├── src/           # Allocation engine and modeling utilities
├── data/          # Processed (non-sensitive) data
├── survey/        # Survey design materials
├── interviews/    # Stakeholder validation notes (appendix material)
└── README.md

🔬 Methodological Components
Representation Learning

PCA for dimensionality reduction

K-means clustering for structural grouping

Latent embedding construction

Structural Compatibility Scoring

Cosine / Euclidean similarity

Apartment centroid computation

Internal dispersion metrics

Ranking of apartment recommendations

Allocation Prototype

Streamlit-based interface

Adjustable scoring weights

Vacancy filters

Demographic rule toggles

⚖ Ethical Considerations

The framework incorporates:

Transparent rule-based demographic constraints

Avoidance of discriminatory nationality rules

Explicit acknowledgment of fairness–diversity trade-offs

Stakeholder interviews to validate synthetic assumptions

The model is designed as decision-support, not automated tenant assignment.