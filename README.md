# Flatmate Group Compatibility Analysis

This repository contains the code, documentation, and structure for a Master’s thesis on predicting **flatmate group compatibility** in co-living environments.  
The project explores whether group-level compatibility can be modelled using survey-based measures of perceived fit, lifestyle preferences, interpersonal expectations, and demographic characteristics.

---

## 📘 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Preprocessing](#preprocessing)
- [Modelling Approach](#modelling-approach)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Data Privacy](#data-privacy)
- [Repository Structure](#repository-structure)
- [Contact](#contact)
- [License](#license)

---

## ⭐ Project Overview

In shared-living settings, tenants frequently report that their satisfaction depends heavily on **how well the flatmate group works together**.  
Despite this, group formation in co-living environments is often random or based on limited information.

This thesis investigates whether **group-level compatibility** can be:

- **measured** (via survey responses),
- **understood** (via feature engineering), and
- **modelled** (via statistical and machine learning approaches)

using the following data sources:

- **Survey-based measures of perceived group fit**  
- **Lifestyle & behavioural preferences**  
- **Expectations toward flatmate interaction**  
- **Demographics**

The goal is not to match individuals pairwise, but to assess whether data-driven insights could support more intentional **group composition** in the future.

---

## 🧩 Dataset

The dataset consists of an anonymous survey completed by tenants of a co-living environment.  
It includes:

- **Self-reported group compatibility / perceived fit**
- **Expectations and values related to shared living**
- **Lifestyle and behavioural preferences** (cleanliness, noise, boundaries, shared activities)
- **Interpersonal dynamics** (comfort, openness, trust)
- **Demographics** (age, nationality, etc.)

No identifiable information is stored in this repository.

---

## 🧱 Features

### **Survey Variables**
- Perceived group fit / “compatibility trust score”
- Lifestyle alignment (e.g., cleanliness routines, noise preferences)
- Personal boundary expectations
- Shared activity preferences
- Attitudes toward communal living
- Conflict tolerance / communication attitudes

### **Demographics**
- Age  
- Nationality  
- Gender identity

### **Derived / Engineered Features**
- Normalised compatibility indicators  
- Group lifestyle cohesion scores  
- Expectation alignment metrics  

---

## 🔧 Preprocessing

Preprocessing steps include:

1. **Cleaning and validating survey responses**
2. **Encoding Likert-scale answers**
3. **Imputing missing values where appropriate**
4. **Scaling continuous variables**
5. **Constructing composite indicators** (e.g., lifestyle cohesion, interpersonal alignment)
6. **Preparing group-level feature aggregates**

All raw data remain local and are not included in this repository.

---

## 🤖 Modelling Approach

Because the survey captures **group-level compatibility**, models focus on predicting **flat-level outcomes**, not pairwise similarity.

Current modelling approaches include:

- **Logistic regression** for categorical group fit outcomes  
- **Random Forests / Gradient Boosted Trees** for non-linear relationships  
- **Regularised linear models** (e.g., LASSO)  
- **Embedding-based or distance-based similarity metrics** for group feature profiles  
- **Group-level clustering** to identify patterns of harmonious vs. incompatible groups  

The goal is to determine which factors most strongly predict **high perceived group compatibility**.

---

## 📊 Evaluation

Models are evaluated using:

- Accuracy and F1-score for categorical compatibility predictions  
- RMSE or MAE for continuous compatibility trust scores  
- Feature importance and SHAP values  
- Cross-validation across multiple flats  
- Sensitivity tests using ablated feature sets (e.g., demographics-only vs. lifestyle-only)

---

## 📈 Results

Results focus on:

- The extent to which compatibility can be predicted using survey data  
- Which behavioural or lifestyle factors are most predictive  
- Patterns that distinguish high-fit vs. low-fit flats  
- Implications for co-living operators and tenant experience  

(A more detailed results summary will be added after final analysis.)

---

## ▶️ Usage

To reproduce the analysis locally:

```bash
git clone https://github.com/JasperAD11/CoLiving-Thesis.git
cd CoLiving-Thesis


