# Financial Statement Fraud Detection

## Table of Contents
- [Financial Statement Fraud Detection](#financial-statement-fraud-detection)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Objectives](#objectives)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [Jupyter Notebooks in the Repository](#jupyter-notebooks-in-the-repository)
  - [Data Description](#data-description)
    - [Data Generation](#data-generation)

## Overview
The **Financial Statement Fraud Detection** project aims to identify fraudulent financial reports using machine learning models and statistical analysis. By leveraging financial ratios, anomaly detection, and advanced models, this project provides actionable insights into detecting potential fraud.  

For this project, the problem is that there is a need to ensure that the information used in the credit risk modelling process is sound by identifying and flagging fraudulent financial statements from companies that have been issued a guarantee from an insurance company. Currently, the methods for identifying these fraudulent financial statements are not reliable, which can lead to negative credit risk modelling and financial losses for the insurance company.

 `Credit risk is the risk that a borrower defaults and does not honor its obligation to service debt.`

Excerpt From: Bart Baesens, Tony van Gestel. “Credit Risk Management: Basic Concepts: financial risk components, rating analysis, models, economic and regulatory capital”. Apple Books. 

## Objectives

1. Develop a statistical (or other) model that predicts/flags fraudulent financial statements
2. Have a detailed analysis of predicted fraud across different buckets (e.g. industry, year, financial type) and it’s correlation with default
3. Develop a fraud indicator or probability that can be used as an extra variable in the credit risk model to improve the accuracy of credit risk assessments. Having a fraud indicator (or probability) will allow us to:
 - Use it as an extra variable in our credit risk model
 - Give us insight into the appropriateness of the data to be used for modelling purposes
4. Provide insights into the appropriateness of the financial data for credit risk modeling purposes.
5. Provide credit underwriters with accurate and reliable information to make informed decisions about the creditworthiness of companies
6. Help insurance companies to identify potential fraud and reduce their exposure to financial losses.
7. Increase the efficiency of the credit risk modeling process by improving the accuracy of the financial data used in the process.
8. Provide possible solution and recommendations for improving financial reporting standards and regulations to prevent financial statement fraud.

## Features
- Preprocessing and cleaning financial data.
- Financial ratio calculation for enhanced feature engineering.
- Statistical analysis methods (e.g., Z-scores, Benford's Law).
- Machine learning models for fraud detection.
- Visualization of trends, correlations, and anomalies.
- Synthetic data generation for testing.

## Project Structure
```
home/
├── data/                  
│   ├── N/A
├── notebooks/              
│   ├── data-cleaning.ipynb            
│   ├── EDA.ipynb
│   ├── feature-engineering.ipynb
│   ├── model-training.ipynb
│   ├── statistical-analysis.ipynb
├── scripts/ 
│   ├── synthetic_data_generator.py
├── src/                 
│   ├── dependencies.py
│   ├── functions.py

```

## Jupyter Notebooks

Explore the `notebooks/` directory for interactive workflows:

# Jupyter Notebooks in the Repository

1. [data-cleaning.ipynb](notebooks/data-cleaning.ipynb) - Cleaning and preprocessing the data.
2. [EDA.ipynb](notebooks/EDA.ipynb) - Performing Exploratory Data Analysis (EDA).
3. [feature-engineering.ipynb](notebooks/feature-engineering.ipynb) - Creating advanced features and performing scaling.
4. [model-training.ipynb](notebooks/model-training.ipynb) - Training and evaluating the machine learning models.
5. [statistical-analysis.ipynb](notebooks/statistical-analysis.ipynb) - Investigating fraudulent patterns using statistical analysis.


---

## Data Description

The dataset used in this project is entirely synthetic and was created to replicate the characteristics of real-world financial data. Key features include:

- **Revenue**: Total revenue of the company, representing its income from operations.
- **Net Income**: Profit after all expenses have been deducted from revenue.
- **Debt to Equity Ratio**: A financial leverage ratio that indicates the relative proportion of debt and equity used to finance a company's assets.
- **Fraud Label**: Binary indicator where:
  - `1`: Financial statement flagged as fraudulent.
  - `0`: Financial statement flagged as non-fraudulent.

### Data Generation
- The synthetic data was generated using advanced statistical methods to ensure:
  - Realistic distributions and correlations between features.
  - Maintenance of patterns observed in real-world financial datasets.
  - Adherence to confidentiality and ethical data usage standards.

Synthetic data ensures the original data is not exposed while providing a suitable foundation for experimentation and analysis.

---
