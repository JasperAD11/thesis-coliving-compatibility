# Structural Compatibility Modeling for Apartment-Level Tenant Allocation in Co-Living

This repository contains the code, documentation, and modelling framework for a Master’s thesis on tenant compatibility in co-living environments.

The project develops a structured compatibility framework for apartment-level tenant allocation. Rather than treating compatibility as a purely subjective or ad hoc judgement, the thesis formalises it through a combination of personality-related fit, lifestyle similarity, and apartment-level structural evaluation in order to support more consistent placement decisions.

## 📘 Project Overview

Commercial co-living environments face recurring challenges related to tenant compatibility, shared routines, and household composition. In practice, allocation decisions are often made with limited structured support and depend heavily on manual judgement.

This thesis develops a two-stage compatibility framework:

### Stage 1 – Personality-Based Compatibility Framework

- Large-scale Big Five personality data
- Construction of trait-based compatibility scores
- Comparison of alternative weighting specifications
- Sensitivity analysis of compatibility assumptions
- Illustrative group formation and assignment simulation

### Stage 2 – Extended Apartment-Level Compatibility Framework

- Integration of personality, lifestyle, and household preference variables
- Use of operational apartment and tenant data
- Applicant-conditioned compatibility modelling
- Apartment-level evaluation based on fairness, harmony, and weakest-pair fit
- Contextual apartment flags for interpretation
- Prototype recommendation interface for vacancy ranking

The goal is not to automate tenant assignment, but to provide a transparent and configurable decision-support framework for co-living operators.

## 🧠 Conceptual Framing

Compatibility is operationalised as a multidimensional and apartment-level concept. The framework combines:

- personality-based interpersonal fit
- lifestyle and household preference similarity
- applicant-specific weighting of compatibility layers
- apartment-level structural evaluation
- contextual information relevant to shared living decisions

The framework emphasizes:

- interpretability
- transparency of scoring logic
- operational relevance for co-living allocation
- decision support rather than automated decision-making

## 📊 Data Sources

### Secondary Data

- OpenPsychometrics Big Five dataset
- Used to develop the Stage 1 personality-based compatibility structure

### Applied Co-Living Data

- Operational tenant and apartment data from a co-living context
- Survey-based lifestyle and household preference variables
- Simulated compatibility-related inputs for structural testing and demonstration

No personally identifiable information is included in this repository.

## 🗂 Repository Structure

```text
CoLiving-Thesis/
│
├── thesis/        # LaTeX source files for the dissertation
├── notebooks/     # Stage 1 and Stage 2 Jupyter notebooks
├── src/           # Compatibility engine and recommendation utilities
├── data/          # Processed (non-sensitive) data
├── survey/        # Survey design materials
└── README.md