# Structural Compatibility Modeling for Apartment-Level Tenant Allocation in Co-Living

This repository contains the code, documentation, and research materials for my Master’s thesis on compatibility modeling in co-living environments.

The project develops a structured framework for apartment-level tenant allocation. Instead of treating compatibility as a purely subjective judgement, it formalizes it through personality-based fit, lifestyle similarity, and apartment-level evaluation to support more consistent and transparent placement decisions.

## Project Overview

Commercial co-living environments face recurring challenges around tenant fit, shared routines, and household composition. In practice, allocation decisions are often made with limited structured support and rely heavily on manual judgement.

This thesis develops a two-stage compatibility framework:

### Stage 1 — Personality-Based Compatibility

- Big Five personality data
- Trait-based pairwise compatibility scoring
- Alternative weighting specifications
- Sensitivity analysis of compatibility assumptions
- Illustrative group formation and assignment simulation

### Stage 2 — Apartment-Level Compatibility

- Integration of personality, lifestyle, and household preference variables
- Use of operational co-living tenant and apartment data
- Applicant-conditioned compatibility modeling
- Apartment-level evaluation using fairness, harmony, and weakest-link fit
- Contextual apartment flags for interpretation
- Prototype recommendation interface for vacancy ranking

The objective is not to automate tenant assignment, but to provide an interpretable and configurable decision-support framework for co-living operators and applicants.

## Conceptual Framing

Compatibility is modeled as a multidimensional and apartment-level concept. The framework combines:

- personality-based interpersonal fit
- lifestyle and household preference similarity
- applicant-specific weighting of compatibility dimensions
- apartment-level structural evaluation
- contextual information relevant to shared living decisions

The framework emphasizes:

- interpretability
- transparent scoring logic
- operational relevance
- decision support rather than automated decision-making

## Data Sources

### Secondary Data
- OpenPsychometrics Big Five dataset  
Used to develop the Stage 1 personality-based compatibility structure.

### Applied Co-Living Data
- operational tenant and apartment data from a co-living context
- survey-based lifestyle and household preference variables
- synthetic inputs used for structural testing and demonstration

No personally identifiable information is included in this repository.

## Repository Structure

```text
thesis-coliving-compatibility/
├── thesis/      # Thesis PDF, references, and figures
├── notebooks/   # Stage 1 and Stage 2 notebooks
├── src/         # Compatibility engine and application components
├── data/        # Processed non-sensitive and synthetic data
├── survey/      # Survey materials
└── README.md
