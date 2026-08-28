# European Insurance Federated Dataset (EIFD) v1.0: A Benchmark for Federated Learning in a Heterogeneous Insurance Domain

## Data Report and Documentation

**Version:** 1.0
**Date:** October 26, 2023
**Authors:** [Your Name/Affiliation]
**Contact:** [Your Email]

---

## Abstract

The European Insurance Federated Dataset (EIFD) v1.0 is a synthetic dataset designed to serve as a benchmark for evaluating federated learning (FL) algorithms in a realistic, heterogeneous, and privacy-sensitive domain. It simulates a scenario where four major European insurance companies—Allianz (Germany), AXA (France), Ageas (Belgium), and Helvetia (Switzerland)—collaboratively train machine learning models without sharing raw customer data. The dataset comprises 20,000 samples, with exactly 5,000 records per client, partitioned into training, validation, and test sets. It includes 21 numerical features, 12 categorical features, and five distinct prediction targets: a regression task (Claim Risk Score), two binary classification tasks (Will Renew, Is High Risk), and a multi-class classification task (Customer Segment). A key attribute of this dataset is its inherent client heterogeneity, as demonstrated by significant variations in target distributions and feature statistics across the four simulated companies. The EIFD is made available in Parquet format alongside a comprehensive set of metadata, data quality reports, and visualizations, providing a robust foundation for reproducible FL research, the development of novel federated algorithms, and comparative studies against centralized benchmarks.

---

## Table of Contents

1. Executive Summary
2. Introduction
   1. Background and Motivation
   2. Research Problem
   3. Research Questions
3. Dataset Overview
   1. Dataset Purpose and Context
   2. Dataset Versioning and Lifecycle
   3. Data Generation and Provenance
   4. Reproducibility and Configuration
4. Federated Dataset Architecture
   1. Federated Clients and Institutional Mapping
   2. The Client-Server Conceptual Model
   3. Data Distribution and Partitioning
5. Dataset Composition
   1. Client-Specific Datasets: Client 1 (Germany)
   2. Client-Specific Datasets: Client 2 (France)
   3. Client-Specific Datasets: Client 3 (Belgium)
   4. Client-Specific Datasets: Client 4 (Switzerland)
   5. The Centralized Dataset: An Aggregated Baseline
6. Data Schema and Data Dictionary
   1. Feature Types and Definitions
   2. Identifier and Metadata Variables
   3. Comprehensive Data Dictionary
7. Target-Variable Analysis
   1. Target 1: Claim Risk Score (Regression)
   2. Target 2: Will Renew (Binary Classification)
   3. Target 3: Is High Risk (Binary Classification)
   4. Target 4: Customer Segment (Multi-class Classification)
8. Statistical Analysis and Client Heterogeneity
   1. Descriptive Statistics of Key Features
   2. Analysis of Client Heterogeneity
   3. Correlation Analysis
9. Data Quality Assessment
   1. Completeness and Missing Value Analysis
   2. Duplicate Analysis
   3. Data Quality Scores
10. Visualization Analysis
    1. Visualization Analysis: Client 1 (Germany)
    2. Visualization Analysis: Client 2 (France)
    3. Visualization Analysis: Client 3 (Belgium)
    4. Visualization Analysis: Client 4 (Switzerland)
    5. Visualization Analysis: Centralized Dataset
11. Federated Learning Implications
    1. Non-IID Characteristics and Statistical Heterogeneity
    2. Label Distribution Skew
    3. Feature Distribution Skew
    4. Privacy and Data Governance Considerations
12. The Centralized Dataset as a Benchmark
13. Machine-Learning Task Formulation
    1. Task 1: Regression of Claim Risk Score
    2. Task 2: Classification of Policy Renewal
    3. Task 3: Classification of High-Risk Customers
    4. Task 4: Multi-class Segmentation
14. Proposed Federated Learning Experiment Design
    1. Baseline Models and Experimental Conditions
    2. Evaluation Framework and Key Metrics
15. Limitations and Threats to Validity
16. Reproducibility and Data Access
17. Conclusion
18. References
19. Appendices
    A. Complete Data Dictionary
    B. Complete Statistical Tables
    C. Figure Catalogue

---

## List of Tables

- **Table 1:** EIFD v1.0 Dataset Overview
- **Table 2:** Federated Client Composition and Institutional Mapping
- **Table 3:** Train/Validation/Test Split Distribution
- **Table 4:** Comprehensive Data Dictionary for EIFD v1.0
- **Table 5:** Target Variable Distribution by Client
- **Table 6:** Central Tendency and Dispersion of Numerical Features
- **Table 7:** Client Heterogeneity Analysis: Key Statistics
- **Table 8:** Client-Level Data Quality Assessment
- **Table 9:** Summary of Missing Values by Client
- **Table 10:** Federated vs. Centralized Dataset Comparison
- **Table 11:** Recommended ML Tasks and Evaluation Metrics

---

## List of Figures

- **Figure 1:** Client Distribution in the Centralized Dataset
- **Figure 2:** Feature Distributions - Centralized Dataset
- **Figure 3:** Target Distributions - Centralized Dataset
- **Figure 4:** Feature Distributions - Client 1 (Germany)
- **Figure 5:** Correlation Matrix - Client 1 (Germany)
- **Figure 6:** Target Distributions - Client 1 (Germany)
- **Figure 7:** Feature Distributions - Client 2 (France)
- **Figure 8:** Correlation Matrix - Client 2 (France)
- **Figure 9:** Target Distributions - Client 2 (France)
- **Figure 10:** Feature Distributions - Client 3 (Belgium)
- **Figure 11:** Correlation Matrix - Client 3 (Belgium)
- **Figure 12:** Target Distributions - Client 3 (Belgium)
- **Figure 13:** Feature Distributions - Client 4 (Switzerland)
- **Figure 14:** Correlation Matrix - Client 4 (Switzerland)
- **Figure 15:** Target Distributions - Client 4 (Switzerland)
- **Figure 16:** Missing Values - Client 2 (France)
- **Figure 17:** Missing Values - Client 1 (Germany)
- **Figure 18:** Missing Values - Client 4 (Switzerland)

---

## 1. Executive Summary

The European Insurance Federated Dataset (EIFD) v1.0 is a synthetic, yet rigorously designed, dataset created to address the critical challenge of developing and evaluating machine learning models in a privacy-preserving, decentralized environment. This dataset simulates a realistic federated learning scenario with four distinct clients representing major European insurance institutions from Germany, France, Belgium, and Switzerland. The primary research problem it aims to solve is how to effectively train models on data that is inherently siloed due to privacy regulations, competitive sensitivities, and institutional data ownership, without the need to centralize sensitive customer information.

To facilitate this, the EIFD provides a balanced distribution of 20,000 total samples, with exactly 5,000 records per client, further divided into standardized training (60%), validation (20%), and test (20%) splits. This uniformity allows for direct and fair comparisons of model performance across clients and learning paradigms. The dataset is composed of a rich set of 33 features (21 numerical and 12 categorical), covering a comprehensive customer and policy profile, alongside five distinct prediction targets that address core business problems in the insurance sector: a regression task for claim risk, a binary classification for policy renewal, a binary classification for high-risk identification, and a multi-class task for customer segmentation.

A central finding from the analysis is the pronounced statistical heterogeneity among the four clients. For instance, the mean Claim Risk Score varies significantly, from a low of 5.20 in Switzerland to a high of 10.18 in France, while the distribution of the "Will Renew" target shows a 7.88 percentage point difference in renewal rates between the best (Switzerland at 94.72%) and worst-performing (France at 86.16%) clients. This heterogeneity, clearly demonstrated by variations in feature distributions and target statistics, makes the EIFD a highly suitable benchmark for testing the robustness of federated learning algorithms to non-IID (non-independent and identically distributed) data, a primary obstacle in practical FL deployments.

Data quality across the clients is generally high, with zero duplicate records and a limited number of missing values, primarily in a few key numerical fields. The data quality scores, derived from the analysis, range from 86.8 (Germany) to 98.8 (Belgium), indicating good overall data integrity. A centralized version of the dataset, created by aggregating all client data, serves as a vital upper-bound benchmark (with a quality score of 95.1) against which the performance of federated models can be compared. This report fully documents the dataset's composition, feature schema, target definitions, inherent heterogeneity, data quality, and its clear utility for machine-learning and federated-learning research, providing a comprehensive foundation for reproducible scientific investigation.

---

## 2. Introduction

### 2.1 Background and Motivation

The modern insurance industry is undergoing a significant transformation driven by the proliferation of data and the adoption of advanced machine learning techniques. Companies are increasingly leveraging data to refine actuarial models, personalize customer experiences, enhance risk assessment, improve fraud detection, and optimize operational efficiency (Bühlmann & Gisler, 2005). However, the data required for these sophisticated models—customer demographics, policy details, claim histories, and financial behavior—is extremely sensitive. This sensitivity gives rise to profound challenges regarding privacy, security, and regulatory compliance.

Traditional approaches to machine learning often operate under the assumption that data can be collected and centralized in a single repository for model training. This paradigm is frequently infeasible in the insurance sector for several reasons. First, stringent data protection regulations, such as the General Data Protection Regulation (GDPR) in Europe, impose strict limitations on the transfer and processing of personal data across borders and between institutions (Voigt & Von dem Bussche, 2017). Second, insurance companies maintain their data as a proprietary asset, representing a significant competitive advantage; they are generally unwilling to share raw customer information with competitors or third parties. Third, even within a single multinational corporation, data silos often exist due to different national subsidiaries operating under independent regulatory frameworks and historical legacy systems. These factors create an environment where data is inherently distributed, leading to a situation described as "data silos" or the "data islands" problem (Kairouz et al., 2021).

### 2.2 Research Problem

The central research problem this dataset addresses is:

**How can machine-learning models be trained collaboratively across geographically and institutionally distributed insurance clients without requiring raw, customer-level data to be centrally transferred or shared?**

This problem is not only of academic interest but also of immense practical and commercial importance. The development of a solution would allow the entire industry to benefit from larger, more diverse datasets, leading to more robust and accurate models without compromising customer privacy or competitive positioning. This challenge sits at the intersection of machine learning, distributed systems, cryptography, and data governance.

Federated Learning (FL) has emerged as the primary and most promising machine learning paradigm for addressing this problem (McMahan et al., 2017). FL enables model training across a network of decentralized clients holding local data samples, without the need to exchange the data itself. Instead, clients compute model updates (e.g., gradient information) based on their local datasets and send only these updates to a central server. The server aggregates the updates (using algorithms like Federated Averaging, or FedAvg) to improve a global model, which is then redistributed to the clients. This process is repeated iteratively until the model converges. FL thus offers a compelling solution by preserving data locality and providing a formal privacy guarantee, as the raw data never leaves the client's infrastructure.

### 2.3 Research Questions

To effectively evaluate the utility of this dataset and the performance of federated learning methods in this insurance context, the following research questions are proposed:

- **RQ1:** How heterogeneous are the insurance datasets across the four federated clients in terms of feature and target distributions, and what are the practical implications for federated optimization algorithms?
- **RQ2:** How do the distributions of key target variables (Claim Risk Score, Will Renew, Is High Risk, and Customer Segment) differ across countries and insurance institutions, and what does this heterogeneity suggest about local market conditions and customer behavior?
- **RQ3:** How does the performance of a centrally trained model on the combined dataset compare to that of a model trained via federated learning? This will quantify the "price of privacy" or the impact of heterogeneity on model performance.
- **RQ4:** To what extent does client-level data quality (e.g., missing data, outliers) affect the performance of both local and federated models?
- **RQ5:** Which target formulation—risk regression, renewal prediction, high-risk classification, or segmentation—provides the most informative and challenging benchmark for evaluating future federated learning algorithms in this domain?

---

## 3. Dataset Overview

### 3.1 Dataset Purpose and Context

The EIFD v1.0 is a purpose-built dataset created to serve as a standardized, reproducible benchmark for academic and industrial research in federated learning. It is designed to emulate the challenges of training models in a realistic, heterogeneous, and privacy-sensitive domain. The dataset allows researchers to investigate, compare, and benchmark various federated learning algorithms (e.g., FedAvg, FedProx, FedMA) in a controlled environment. It also provides a common foundation for studying related problems such as client selection, differential privacy, secure aggregation, and the effects of data non-IID-ness. The dataset is explicitly designed to represent the four of the largest European insurance markets and some of their most prominent companies, thus providing a contextually grounded and meaningful simulation.

### 3.2 Dataset Versioning and Lifecycle

This report documents the initial release of the dataset, designated as version **v1.0**. Future versions may include updates such as additional clients, an expanded feature set, different data distribution patterns (e.g., varying sample counts per client), or modifications to the underlying data generation process to better reflect real-world complexities. All subsequent releases will be clearly versioned and accompanied by detailed documentation outlining the changes made to ensure transparent and reproducible research.

### 3.3 Data Generation and Provenance

The dataset is synthetic, generated by a Python-based generator script (`data_generator.py`). The synthetic nature of the data allows for the controlled introduction of specific properties (like controlled statistical heterogeneity) and ensures that the complete population parameters are known, which is often not possible with real-world data. The generation process is based on real-world country-level statistics, such as income, savings rates, claim frequencies, life expectancy, and risk appetite, as defined in the `country_stats` dictionary within the generator. This anchoring in real-world statistics ensures that the simulated data exhibits believable and interpretable patterns. For instance, the generator uses real-world values for average income, savings rates, and claim frequencies for each of the four countries to produce samples with realistic correlations. Each client's data is then generated independently, which naturally creates a non-IID distribution that is key to FL research.

### 3.4 Reproducibility and Configuration

A key design goal of the EIFD is full reproducibility. The entire dataset was generated using a fixed random seed of `42`, ensuring that the exact same dataset can be recreated at any time. The primary software environment and key dependencies are listed in Table 1, ensuring that the computational environment is clearly defined. The dataset is saved in the Apache Parquet format, a columnar storage format that offers high performance for data analysis and ML workloads. The comprehensive metadata files (`metadata.json`, `dataset_info.json`) and the quality reports produced by the `analyze_data.py` script are an integral part of the dataset, providing researchers with immediate insight into its structure and properties.

**Table 1: EIFD v1.0 Dataset Overview**

| Property | Value |
| :--- | :--- |
| **Dataset Name** | European Insurance Federated Dataset |
| **Version** | v1.0 |
| **Framework** | PyTorch |
| **Random Seed** | 42 |
| **Format** | Parquet |
| **Number of Clients** | 4 |
| **Samples per Client** | 5,000 |
| **Total Samples** | 20,000 |
| **Centralized Dataset** | Yes (20,000 samples) |
| **Python Version** | 3.11.16 |
| **Key Dependencies** | NumPy, Pandas, PyTorch, Scikit-learn, Matplotlib, Seaborn, PyArrow |

---

## 4. Federated Dataset Architecture

### 4.1 Federated Clients and Institutional Mapping

The EIFD consists of four federated clients, each representing a major European insurance institution. The mapping between the client identifier, its country, and the insurance company is provided in Table 2. This institutional mapping adds realism and allows for interpretations based on the known characteristics of these markets. It is important to note that while these institutions are named for contextual grounding, the data itself is entirely synthetic and does not represent actual customer information from these companies.

**Table 2: Federated Client Composition and Institutional Mapping**

| Client Identifier | Country | Simulated Institution |
| :--- | :--- | :--- |
| `germany_allianz` | Germany | Allianz (simulated) |
| `france_axa` | France | AXA (simulated) |
| `belgium_ageas` | Belgium | Ageas (simulated) |
| `switzerland_helvetia` | Switzerland | Helvetia (simulated) |

### 4.2 The Client-Server Conceptual Model

The underlying architecture of the federated learning process is conceptually straightforward. A central server orchestrates the training without having direct access to the raw data. The process is as follows:

- The server initializes a global model and sends its parameters to all participating clients.
- Each client trains the model using its own local, private dataset for several local epochs.
- Clients then send their model updates back to the server.
- The server aggregates these updates (e.g., by performing a weighted average) to create an improved global model.
- This loop of distribution, local training, and aggregation is repeated for numerous communication rounds until the global model converges.

This model perfectly mirrors the structure of the provided federated dataset, where each client's data is physically isolated.

### 4.3 Data Distribution and Partitioning

The dataset is split both in terms of client ownership and in terms of training, validation, and testing sets. This is a critical design choice for conducting rigorous ML experiments. Each client's data is partitioned as follows:

- **Training Set:** 60% of the client's data (3,000 samples). This is the data used for local model training.
- **Validation Set:** 20% of the client's data (1,000 samples). This is used for hyperparameter tuning and early stopping during local training.
- **Test Set:** 20% of the client's data (1,000 samples). This "held-out" data is used exclusively for final evaluation, providing an unbiased estimate of the model's generalization performance.

The distribution of these splits across all clients is shown in Table 3. The consistent split ratio across clients is essential for ensuring fair comparisons when evaluating the performance of different FL algorithms.

**Table 3: Train/Validation/Test Split Distribution**

| Client Identifier | Total Samples | Train Samples | Validation Samples | Test Samples |
| :--- | :--- | :--- | :--- | :--- |
| germany_allianz | 5,000 | 3,000 | 1,000 | 1,000 |
| france_axa | 5,000 | 3,000 | 1,000 | 1,000 |
| belgium_ageas | 5,000 | 3,000 | 1,000 | 1,000 |
| switzerland_helvetia | 5,000 | 3,000 | 1,000 | 1,000 |
| **Centralized** | **20,000** | **12,000** | **4,000** | **4,000** |

---

## 5. Dataset Composition

This section details the composition of each client-specific dataset and the centralized dataset, drawing heavily on the exact counts and statistics provided in the generation log and metadata.

### 5.1 Client-Specific Datasets: Client 1 (Germany)

- **Samples:** 5,000
- **Train/Validation/Test:** 3,000/1,000/1,000
- **Features:** 46
- **Quality Score:** 86.8/100
- **Target Statistics:**
    - **Claim Risk Score:** 8.44 ± 6.55 (Mean ± Std Dev)
    - **Will Renew:** 4,723 (94.46%) renew, 277 (5.54%) not renew.
    - **Is High Risk:** 3,240 (64.8%) not high-risk, 1,760 (35.2%) high-risk.
    - **Customer Segment:** Segment 1: 1,761 (35.22%), Segment 2: 1,716 (34.32%), Segment 0: 1,523 (30.46%).

### 5.2 Client-Specific Datasets: Client 2 (France)

- **Samples:** 5,000
- **Train/Validation/Test:** 3,000/1,000/1,000
- **Features:** 46
- **Quality Score:** 88.9/100
- **Target Statistics:**
    - **Claim Risk Score:** 10.18 ± 8.28
    - **Will Renew:** 4,308 (86.16%) renew, 692 (13.84%) not renew.
    - **Is High Risk:** 3,275 (65.5%) not high-risk, 1,725 (34.5%) high-risk.
    - **Customer Segment:** Segment 0: 1,766 (35.32%), Segment 2: 1,740 (34.8%), Segment 1: 1,494 (29.88%).

### 5.3 Client-Specific Datasets: Client 3 (Belgium)

- **Samples:** 5,000
- **Train/Validation/Test:** 3,000/1,000/1,000
- **Features:** 46
- **Quality Score:** 98.8/100
- **Target Statistics:**
    - **Claim Risk Score:** 6.69 ± 5.97
    - **Will Renew:** 4,669 (93.38%) renew, 331 (6.62%) not renew.
    - **Is High Risk:** 3,218 (64.36%) not high-risk, 1,782 (35.64%) high-risk.
    - **Customer Segment:** Segment 0: 1,930 (38.6%), Segment 1: 1,573 (31.46%), Segment 2: 1,497 (29.94%).

### 5.4 Client-Specific Datasets: Client 4 (Switzerland)

- **Samples:** 5,000
- **Train/Validation/Test:** 3,000/1,000/1,000
- **Features:** 46
- **Quality Score:** 90.1/100
- **Target Statistics:**
    - **Claim Risk Score:** 5.20 ± 4.93
    - **Will Renew:** 4,736 (94.72%) renew, 264 (5.28%) not renew.
    - **Is High Risk:** 3,467 (69.34%) not high-risk, 1,533 (30.66%) high-risk.
    - **Customer Segment:** Segment 2: 2,047 (40.94%), Segment 1: 1,751 (35.02%), Segment 0: 1,202 (24.04%).

### 5.5 The Centralized Dataset: An Aggregated Baseline

The centralized dataset is a combination of all four client-specific datasets. It serves as the "complete" dataset and is fundamental for establishing a performance upper-bound for centralized machine learning models.

- **Samples:** 20,000
- **Train/Validation/Test:** 12,000/4,000/4,000
- **Features:** 47
- **Quality Score:** 95.1/100
- **Client Distribution:** Each of the four clients contributes exactly 5,000 samples (25% each) to the centralized dataset. This uniform contribution is shown in Figure 1.

This benchmark is essential for answering RQ3: quantifying the performance gap between centralized and federated learning.

---

## 6. Data Schema and Data Dictionary

### 6.1 Feature Types and Definitions

The EIFD contains a total of 33 features, categorized as follows:

1.  **Numerical Features (21):** These are continuous or integer-valued variables representing financial and personal metrics.
2.  **Categorical Features (12):** These are nominal variables representing demographic and policy attributes.
3.  **Identifier Variables (2):** These are unique identifiers for each customer and their client of origin.
4.  **Metadata Variables (4):** These variables describe the generating context and are not used as inputs for prediction.

The complete data dictionary is provided in Table 4. All numerical features' definitions are derived directly from the data generator and are confirmed by the metadata. It is crucial to note that while client datasets have 46 features, the centralized dataset has 47. This one-feature difference is a key schema discrepancy to be investigated and resolved in future versions.

**Table 4: Comprehensive Data Dictionary for EIFD v1.0**

| Variable | Type | Role | Description | Unit/Scale | Missing % | Cardinality | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Numerical Features** | | | | | | | |
| annual_income | Float64 | Input | Annual income of the customer. | EUR/CHF | 1.535%* | - | Range: 15,000 – 300,000+ |
| savings_balance | Float64 | Input | Total savings balance of the customer. | EUR/CHF | 1.44%* | - | Range: 0 – 200,000+ |
| policy_count | Int64 | Input | Number of policies the customer holds. | count | 0% | 1-12 | - |
| policy_value | Float64 | Input | Total value of all policies held. | EUR/CHF | 0% | - | Range: 5,000 – 5,000,000+ |
| age | Int64 | Input | Customer's age. | years | 0% | 18-95 | - |
| years_as_customer | Int64 | Input | Number of years the customer has been with the company. | years | 0% | 0-54 | - |
| number_of_claims | Int64 | Input | Total number of claims filed by the customer. | count | 0% | 0-4 | - |
| total_claim_amount | Float64 | Input | Total monetary amount of all claims filed. | EUR/CHF | 0% | - | Range: 0 – 190,000+ |
| claim_frequency | Float64 | Input | Number of claims per year of tenure. | claims/year | 0% | 0-0.5 | - |
| property_value | Float64 | Input | Estimated value of the customer's primary property. | EUR/CHF | 0% | - | Range: 100,000 – 1,200,000+ |
| medical_expenses | Float64 | Input | Annual medical expenses. | EUR/CHF | 2.405%* | - | Range: 16 – 30,000+ |
| life_expectancy | Float64 | Input | Estimated life expectancy. | years | 0% | 60-95 | - |
| health_risk_score | Float64 | Input | A health risk assessment score. | score (0-100) | 0% | 0-100 | - |
| driving_experience | Int64 | Input | Years of driving experience. | years | 0% | 0-91 | - |
| vehicle_value | Float64 | Input | Estimated value of the customer's primary vehicle. | EUR/CHF | 0% | - | Range: 5,000 – 190,000+ |
| liability_risk | Float64 | Input | A liability risk score. | score (0-100) | 0% | 0-100 | - |
| debt_to_income_ratio | Float64 | Input | Customer's debt-to-income ratio. | ratio | 0% | 0.05-0.8 | - |
| credit_score | Int64 | Input | Customer's credit score. | score (300-850) | 0% | 300-850 | - |
| family_members | Int64 | Input | Number of family members. | count | 0% | 1-8 | - |
| policy_duration | Int64 | Input | Duration of the current policy in years. | years | 0% | 1-30 | - |
| premium_amount | Float64 | Input | Annual premium amount paid by the customer. | EUR/CHF | 0% | 200-20,000+ | - |
| **Categorical Features** | | | | | | | |
| marital_status | Object | Input | Marital status of the customer. | - | 0% | 4 | Single, Married, Divorced, Widowed |
| education_level | Object | Input | Highest level of education attained. | - | 0% | 5 | Secondary, Bachelor, Master, PhD, Professional |
| employment_status | Object | Input | Employment status. | - | 0% | 4 | Employed, Self-Employed, Unemployed, Retired |
| home_ownership | Object | Input | Home ownership status. | - | 0% | 4 | Own, Rent, Mortgage, Other |
| insurance_package | Object | Input | Type of insurance package. | - | 0% | 4 | Basic, Standard, Premium, Elite |
| risk_category | Object | Input | A categorical risk assessment. | - | 0% | 4 | Low, Medium, High, Very High |
| policy_type | Object | Input | Type of policy held. | - | 0% | 3 | Individual, Family, Corporate |
| vehicle_type | Object | Input | Type of vehicle. | - | 0% | 5 | Sedan, SUV, Sports, Truck, Motorcycle |
| property_type | Object | Input | Type of property. | - | 0% | 4 | Apartment, House, Villa, Commercial |
| profession | Object | Input | Customer's profession. | - | 0% | 6 | Engineer, Doctor, Lawyer, Teacher, Business, Other |
| primary_insurance_type | Object | Input | The primary insurance product type. | - | 0% | 7 | Auto, Life, Health, Property, Liability, Travel, Accident |
| policy_status | Object | Input | Current status of the policy. | - | 0% | 4 | Active, Lapsed, Cancelled, Pending |
| **Identifier Variables** | | | | | | | |
| customer_id | Object | Metadata | A unique identifier for each customer. | - | 0% | 20,000 | - |
| client_id | Object | Metadata | A unique identifier for the originating client. | - | 0% | 4 | Added to centralized set. |
| **Target Variables** | | | | | | | |
| claim_risk_score | Float64 | Target | Predicted risk score for future claims. | score (0-100) | 0% | - | Regression target. |
| customer_lifetime_value | Float64 | Target | Predicted total value of the customer. | EUR/CHF | 0% | - | Regression target. |
| will_renew | Int64 | Target | Binary indicator of policy renewal. | 0/1 | 0% | 2 | Binary classification target. |
| is_high_risk | Int64 | Target | Binary indicator of being a high-risk customer. | 0/1 | 0% | 2 | Binary classification target. |
| customer_segment | Int64 | Target | Customer segment label. | 0/1/2 | 0% | 3 | Multi-class target. |
| \*Note: Missing percentages are for the centralized dataset and may vary per client. | | | | | | | |

---

## 7. Target-Variable Analysis

The EIFD provides four distinct and realistic prediction targets, each addressing a different facet of insurance analytics. This section details their distributions and defines the research tasks they support.

### 7.1 Target 1: Claim Risk Score (Regression)

- **Definition:** The `claim_risk_score` is a continuous variable designed to represent the expected risk of a customer filing a claim in the future. A higher score indicates a higher predicted risk. This target is fundamental to actuarial modeling and underwriting.
- **Analysis:** The mean score shows significant client heterogeneity. It is highest for France (10.18 ± 8.28), suggesting a more risky portfolio, and lowest for Switzerland (5.20 ± 4.93), suggesting a lower-risk portfolio. This aligns with the source statistics where France has a higher real-world claim frequency (0.055) than Switzerland (0.030).
- **Implications:** This target is a classic regression task. Differences in its distribution across clients will cause a significant label distribution skew in FL, making it a prime test for algorithms designed for regression with non-IID data.

### 7.2 Target 2: Will Renew (Binary Classification)

- **Definition:** This binary target predicts whether a customer will renew their policy (`1`) or not (`0`). Customer retention is a critical business metric.
- **Analysis:** The renewal rates are extremely high in this simulated dataset, ranging from 86.16% (France) to 94.72% (Switzerland). This creates a class imbalance problem (see Table 5).
- **Implications:** This task is a binary classification problem where the primary challenge is the class imbalance, which is further complicated by the variations in imbalance ratio across clients (e.g., Switzerland's ratio is 17.94, while France's is 6.23).

### 7.3 Target 3: Is High Risk (Binary Classification)

- **Definition:** This target predicts whether a customer is categorized as "high risk" (`1`) or not (`0`). This likely combines elements of claim risk, credit risk, and liability risk.
- **Analysis:** The distribution is relatively balanced across all clients, with the "high risk" proportion ranging from 30.66% (Switzerland) to 35.64% (Belgium). This balance is beneficial for standard classification algorithms and FL.
- **Implications:** This represents a moderately imbalanced binary classification task that is relatively consistent across clients, making it a good candidate for evaluating the robustness of FL to changes in feature distributions rather than label distributions.

### 7.4 Target 4: Customer Segment (Multi-class Classification)

- **Definition:** This target assigns each customer to one of three segments (0, 1, 2). Based on the generation logic, these segments correspond to "Standard" (0), "Premium" (1), and "Elite" (2) customers, likely reflecting their value and potential.
- **Analysis:** The segment distribution is relatively balanced, though not perfectly uniform. Notable variations exist, such as Switzerland having a significantly higher proportion of "Elite" customers (40.94%) compared to Belgium (29.94%).
- **Implications:** This is a 3-class classification problem with a moderate level of class balance. It allows for the evaluation of FL algorithms on a multi-class setting with both label and feature distribution skew.

**Table 5: Target Variable Distribution by Client**

| Target | Metric | Germany | France | Belgium | Switzerland |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Claim Risk** | Mean ± Std | 8.44 ± 6.55 | 10.18 ± 8.28 | 6.69 ± 5.97 | 5.20 ± 4.93 |
| **Will Renew** | Renew (1) | 4,723 (94.46%) | 4,308 (86.16%) | 4,669 (93.38%) | 4,736 (94.72%) |
| | Not Renew (0) | 277 (5.54%) | 692 (13.84%) | 331 (6.62%) | 264 (5.28%) |
| | Imbalance Ratio | 17.05 | 6.23 | 14.11 | 17.94 |
| **Is High Risk** | Not High Risk (0) | 3,240 (64.80%) | 3,275 (65.50%) | 3,218 (64.36%) | 3,467 (69.34%) |
| | High Risk (1) | 1,760 (35.20%) | 1,725 (34.50%) | 1,782 (35.64%) | 1,533 (30.66%) |
| | Imbalance Ratio | 1.84 | 1.90 | 1.81 | 2.26 |
| **Customer Segment** | Segment 0 | 1,523 (30.46%) | 1,766 (35.32%) | 1,930 (38.60%) | 1,202 (24.04%) |
| | Segment 1 | 1,761 (35.22%) | 1,494 (29.88%) | 1,573 (31.46%) | 1,751 (35.02%) |
| | Segment 2 | 1,716 (34.32%) | 1,740 (34.80%) | 1,497 (29.94%) | 2,047 (40.94%) |

---

## 8. Statistical Analysis and Client Heterogeneity

### 8.1 Descriptive Statistics of Key Features

The strong heterogeneity between clients is evident across most numerical features. For example, the mean annual income varies considerably between Switzerland (€65,267) and Belgium (€42,082), while the average life expectancy is highest in Switzerland (83.93 years) and similar in other countries. These differences are a direct result of the country-specific statistics used in the data generation, as shown in Table 6.

**Table 6: Central Tendency and Dispersion of Numerical Features**

| Feature | Germany (Mean ± Std) | France (Mean ± Std) | Belgium (Mean ± Std) | Switzerland (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- |
| annual_income | 49,835 ± 16,492 | 48,684 ± 41,338 | 42,082 ± 13,874 | 65,268 ± 22,366 |
| savings_balance | 3,555 ± 1,413 | 2,856 ± 1,158 | 2,783 ± 1,113 | 5,971 ± 2,396 |
| policy_value | 130,485 ± 71,719 | 117,347 ± 64,886 | 111,062 ± 60,160 | 171,897 ± 95,959 |
| age | 46.26 ± 14.51 | 42.14 ± 14.17 | 42.02 ± 14.45 | 43.18 ± 14.51 |
| number_of_claims | 0.29 ± 0.54 | 0.34 ± 0.58 | 0.24 ± 0.49 | 0.20 ± 0.45 |
| credit_score | 682.79 ± 56.15 | 675.47 ± 56.38 | 670.56 ± 54.99 | 710.03 ± 61.55 |

### 8.2 Analysis of Client Heterogeneity

The heterogeneity analysis (Table 7) reveals systematic differences between the simulated clients.

1.  **Label Distribution Skew:** The most significant evidence of heterogeneity is in the target variables. The mean Claim Risk Score varies by almost 100% between France (10.18) and Switzerland (5.20). Similarly, the renewal rate is nearly 9 percentage points higher in Switzerland than in France. This non-IID nature of the labels is a core challenge in FL.
2.  **Feature Distribution Skew:** The clients also exhibit clear covariate shift. For instance, the mean annual income in Switzerland is over 50% higher than in Belgium. This indicates that models trained on one client's data will likely perform poorly when applied directly to another client's data, a phenomenon known as domain adaptation.
3.  **Interpretation and Implications:** This distributional heterogeneity has several critical implications for FL. First, the global model obtained by averaging local model updates may underperform for specific clients if the heterogeneity is not addressed by the algorithm. This is the core problem of personalization in FL. Second, standard evaluation metrics like global average accuracy can be misleading; per-client performance and fairness across clients become paramount. Finally, this dataset provides an excellent testbed for algorithms explicitly designed to handle heterogeneity, such as those incorporating proximal terms (e.g., FedProx) or dynamic client weighting.

**Table 7: Client Heterogeneity Analysis: Key Statistics**

| Metric | Germany | France | Belgium | Switzerland |
| :--- | :--- | :--- | :--- | :--- |
| **Samples** | 5,000 | 5,000 | 5,000 | 5,000 |
| **Quality Score** | 86.8 | 88.9 | 98.8 | 90.1 |
| **Missing Columns** | 2 | 1 | 0 | 1 |
| **Claim Risk Mean** | 8.44 | 10.18 | 6.69 | 5.20 |
| **Claim Risk SD** | 6.55 | 8.28 | 5.97 | 4.93 |
| **Renewal Rate (%)** | 94.46 | 86.16 | 93.38 | 94.72 |
| **High Risk Rate (%)** | 35.20 | 34.50 | 35.64 | 30.66 |
| **Mean Income (EUR)** | 49,835 | 48,684 | 42,082 | 65,268 |

### 8.3 Correlation Analysis

The correlation plots for each client provide insight into the underlying data-generating process. Key observations include:

- **Strong Positive Correlations:** As expected, `policy_value` and `premium_amount` are strongly correlated (e.g., 0.71 in Germany, 0.59 in France). Similarly, `age` and `driving_experience` are highly correlated (e.g., 0.68 in Germany, 0.72 in France). These correlations are consistent across clients and reflect realistic relationships.
- **Interpreting Feature-Target Correlations:** The correlation matrices show that `number_of_claims`, `claim_frequency`, and `total_claim_amount` are most strongly correlated with the target `claim_risk_score`. This confirms that the target was generated logically from its constituent features.
- **Multicollinearity:** A few feature pairs show very high correlations, such as `policy_value` and `premium_amount`, and `age` and `driving_experience`. This suggests that one of the features in these highly correlated pairs could be dropped to reduce multicollinearity without significant loss of information, simplifying the model.

---

## 9. Data Quality Assessment

A comprehensive data quality assessment was performed, yielding the metrics in Table 8.

**Table 8: Client-Level Data Quality Assessment**

| Metric | Germany | France | Belgium | Switzerland | Centralized |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Quality Score** | 86.8 | 88.9 | 98.8 | 90.1 | 95.1 |
| **Columns with Missing Values** | 2 | 1 | 0 | 1 | 3 |
| **Duplicate Rate** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### 9.1 Completeness and Missing Value Analysis

The absence of duplicates is a positive indicator of data integrity. The missing value analysis reveals that only a small number of columns and records contain null values.

- **Germany:** Missing values in `annual_income` (6.14%) and `savings_balance` (5.76%).
- **France:** Missing values in `medical_expenses` (5.14%).
- **Switzerland:** Missing values in `medical_expenses` (4.48%).
- **Centralized:** Missing values in `annual_income` (1.535%), `savings_balance` (1.44%), and `medical_expenses` (2.405%).

The mechanism of missingness is not documented. It is likely a form of missing at random (MAR) or missing completely at random (MCAR) introduced for realism. However, this cannot be confirmed without analyzing other variables. These missing values can be addressed through common imputation techniques like mean/median imputation, KNN imputation, or using models that can handle missing data natively (e.g., XGBoost).

### 9.2 Duplicate Analysis

The duplicate rate for all clients and the centralized dataset is reported as **0.00%**. The absence of duplicate records is a desirable property for any dataset used in supervised learning, as duplicates can artificially inflate the importance of certain observations and lead to biased model training.

### 9.3 Data Quality Scores

The overall quality scores are high, with a mean of 91.15 across the four clients. The centralized dataset has the highest score of 95.1. The specific components of this score are not public, but it likely accounts for a combination of data completeness, outlier presence, and feature consistency. Germany's lower score (86.8) is likely driven by its higher number of missing values, while Belgium's perfect score reflects its complete dataset.

---

## 10. Visualization Analysis

### 10.1 Visualization Analysis: Client 1 (Germany)

![Feature Distributions - Client 1 (Germany)](outputs/reports/analysis/clients/germany_allianz/plots/distributions_germany_allianz.png)

**Figure 4: Feature Distributions - Client 1 (Germany)**

The histograms above show the distribution of the first 12 numerical features for the German client. Many features, such as `annual_income` and `savings_balance`, appear roughly normally distributed. However, features like `number_of_claims` and `claim_frequency` are highly right-skewed, as the vast majority of customers have zero claims. This skewness is a common characteristic of insurance data and should be accounted for in modeling.

![Correlation Matrix - Client 1 (Germany)](outputs/reports/analysis/clients/germany_allianz/plots/correlation_germany_allianz.png)

**Figure 5: Correlation Matrix - Client 1 (Germany)**

This plot confirms the strong, positive correlations discussed in Section 8.3. The correlations between features like `policy_value` and `premium_amount` are clearly visible, as are the relationships between `age`, `years_as_customer`, and `driving_experience`. The relatively high correlations between some input features suggest potential multicollinearity.

![Target Distributions - Client 1 (Germany)](outputs/reports/analysis/clients/germany_allianz/plots/targets_germany_allianz.png)

**Figure 6: Target Distributions - Client 1 (Germany)**

The targets show the expected distributions. `claim_risk_score` is right-skewed, `will_renew` is heavily imbalanced in favor of renewals, `is_high_risk` is relatively balanced, and `customer_segment` is almost uniformly distributed. These patterns highlight the different challenges posed by each prediction task.

### 10.2 Visualization Analysis: Client 2 (France)

![Feature Distributions - Client 2 (France)](outputs/reports/analysis/clients/france_axa/plots/distributions_france_axa.png)

**Figure 7: Feature Distributions - Client 2 (France)**

The feature distributions for France are similar in shape to Germany's but exhibit different means. The distributions for `annual_income` and `savings_balance` appear slightly more variable than in Germany, potentially reflecting a more diverse customer base.

![Correlation Matrix - Client 2 (France)](outputs/reports/analysis/clients/france_axa/plots/correlation_france_axa.png)

**Figure 8: Correlation Matrix - Client 2 (France)**

The correlation structure is very similar to Germany's, with high correlations between `policy_value`-`premium_amount` and `age`-`driving_experience`. This consistency across clients suggests a stable data-generating mechanism.

![Target Distributions - Client 2 (France)](outputs/reports/analysis/clients/france_axa/plots/targets_france_axa.png)

**Figure 9: Target Distributions - Client 2 (France)**

The most notable difference is in the `will_renew` target, which shows a lower renewal rate (more zeros) than the other clients, reflecting the class imbalance. The `claim_risk_score` also appears to have a higher mean than in Germany or Switzerland.

### 10.3 Visualization Analysis: Client 3 (Belgium)

![Feature Distributions - Client 3 (Belgium)](outputs/reports/analysis/clients/belgium_ageas/plots/distributions_belgium_ageas.png)

**Figure 10: Feature Distributions - Client 3 (Belgium)**

The feature distributions are consistent with the lower income profile of Belgium. The shapes are similar to the other clients, but the central tendencies are lower for income-related features.

![Correlation Matrix - Client 3 (Belgium)](outputs/reports/analysis/clients/belgium_ageas/plots/correlation_belgium_ageas.png)

**Figure 11: Correlation Matrix - Client 3 (Belgium)**

The correlation structure is consistent with the other clients, further confirming the stability of the relationships between features.

![Target Distributions - Client 3 (Belgium)](outputs/reports/analysis/clients/belgium_ageas/plots/targets_belgium_ageas.png)

**Figure 12: Target Distributions - Client 3 (Belgium)**

The target distributions are consistent with the other clients, with a high but slightly lower renewal rate than Germany and Switzerland. The `claim_risk_score` distribution shows a lower mean than France but higher than Switzerland.

### 10.4 Visualization Analysis: Client 4 (Switzerland)

![Feature Distributions - Client 4 (Switzerland)](outputs/reports/analysis/clients/switzerland_helvetia/plots/distributions_switzerland_helvetia.png)

**Figure 13: Feature Distributions - Client 4 (Switzerland)**

The feature distributions are notably different, with visibly higher means for features like `annual_income` and `savings_balance`, reflecting the higher income profile of Switzerland. This is a clear example of covariate shift.

![Correlation Matrix - Client 4 (Switzerland)](outputs/reports/analysis/clients/switzerland_helvetia/plots/correlation_switzerland_helvetia.png)

**Figure 14: Correlation Matrix - Client 4 (Switzerland)**

The correlation structure is consistent across all clients, suggesting that the relationships between features are stable even as their distributions shift.

![Target Distributions - Client 4 (Switzerland)](outputs/reports/analysis/clients/switzerland_helvetia/plots/targets_switzerland_helvetia.png)

**Figure 15: Target Distributions - Client 4 (Switzerland)**

The target distributions show the lowest `claim_risk_score` and `is_high_risk` rates, aligning with the lower-risk profile of Switzerland. The `customer_segment` distribution is also the most skewed in favor of Segment 2, suggesting a higher proportion of high-value customers.

### 10.5 Visualization Analysis: Centralized Dataset

![Client Distribution in Centralized Dataset](outputs/reports/analysis/centralized/plots/centralized_client_distribution.png)

**Figure 1: Client Distribution in Centralized Dataset**

This confirms the dataset is perfectly balanced, with exactly 5,000 samples from each client. This ensures any performance discrepancies in a centralized model cannot be attributed to a class imbalance in terms of client origin.

![Feature Distributions - Centralized Dataset](outputs/reports/analysis/centralized/plots/centralized_distributions.png)

**Figure 2: Feature Distributions - Centralized Dataset**

This figure shows the aggregate distributions. The overall shapes are a mixture of the individual client distributions, effectively smoothing out the client-level heterogeneity.

![Target Distributions - Centralized Dataset](outputs/reports/analysis/centralized/plots/centralized_targets.png)

**Figure 3: Target Distributions - Centralized Dataset**

The centralized target distributions are a weighted average of the individual client distributions. This highlights the smoothing effect of centralizing data, which can mask the underlying heterogeneity that makes FL challenging.

---

## 11. Federated Learning Implications

### 11.1 Non-IID Characteristics and Statistical Heterogeneity

The analysis confirms the dataset's non-IID nature at multiple levels. There is clear **label distribution skew**, as evidenced by the differing means and class balances of the target variables across clients (Table 5). There is also clear **feature distribution skew**, as shown by the different means and variances of features like `annual_income` (Table 6). This heterogeneity presents a significant challenge for algorithms like FedAvg, which implicitly assume the data is independent and identically distributed (IID).

### 11.2 Label Distribution Skew

The variations in label distributions will cause the global model to be biased towards the majority label or the most common distribution. In a scenario where one client has a much lower renewal rate (France), the global model might underperform on their data. This is a classic problem in FL and can be addressed by algorithms that are robust to label skew, such as those using personalized aggregation or dynamic weighting.

### 11.3 Feature Distribution Skew

The covariate shift between clients means that the learned feature representations may not generalize well from one client to another. Personalization techniques, such as local fine-tuning (Fine-tune) or multi-task learning, will likely be needed to achieve optimal performance on each client's local data. This is a key area for future research using this dataset.

### 11.4 Privacy and Data Governance Considerations

The federated architecture is a powerful privacy-enhancing technology (PET), but it is crucial to contextualize its role. While the raw data remains on the client's side, FL is not inherently cryptographically private. It still exposes information through model updates (gradients). This dataset provides the perfect infrastructure for evaluating privacy-preserving techniques in a realistic setting. For instance, researchers can integrate the EIFD with tools for:

- **Differential Privacy (DP):** Adding noise to model updates to provide a mathematical guarantee of privacy.
- **Secure Aggregation (SecAgg):** Encrypting model updates so the server only learns the aggregated result, not individual client updates.
- **Homomorphic Encryption (HE):** Performing computations on encrypted data.

---

## 12. The Centralized Dataset as a Benchmark

The centralized dataset is a critical component of the EIFD. It allows researchers to answer RQ3 and establish a performance "upper bound" for their models. By training the same model architecture on the centralized data and comparing its performance to the federated global model, researchers can quantify the loss in performance incurred due to the federated process and the data heterogeneity. This is essential for evaluating the trade-off between privacy and model accuracy. The centralized dataset also simplifies model development and tuning, as it allows for a "gold standard" reference to be established before tackling the complexities of FL.

**Table 10: Federated vs. Centralized Dataset Comparison**

| Property | Federated | Centralized |
| :--- | :--- | :--- |
| Raw Data Location | Distributed (4 clients) | Combined in a single dataset |
| Dataset Splits | 12,000/4,000/4,000 | 12,000/4,000/4,000 |
| Privacy Motivation | High | Lower |
| Client Heterogeneity | Preserved | Aggregated / Smoothed |
| Primary Use | FL algorithm evaluation | Centralized benchmark |

---

## 13. Machine-Learning Task Formulation

The EIFD is designed for four distinct tasks (Table 11).

**Table 11: Recommended ML Tasks and Evaluation Metrics**

| Task | Target | Type | Input Features | Output | Objective | Key Evaluation Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `claim_risk_score` | Regression | All Inputs | Continuous score | Predict future claim risk | MAE, MSE, RMSE, R² |
| **2** | `will_renew` | Binary Classification | All Inputs | 0/1 | Predict policy renewal | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC (imbalance aware) |
| **3** | `is_high_risk` | Binary Classification | All Inputs | 0/1 | Identify high-risk customers | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| **4** | `customer_segment` | Multi-class Classification | All Inputs | 0, 1, or 2 | Assign customers to segments | Accuracy, Macro/Micro F1, Confusion Matrix |

---

## 14. Proposed Federated Learning Experiment Design

A sound experimental design is crucial for evaluating FL algorithms. A comprehensive framework is outlined below.

**1. Baseline Models:**

- **Local-only Model:** Train a separate model for each client, using only its local training data. Performance variance highlights data heterogeneity.
- **Centralized Model:** Train a single model on the entire centralized dataset. This is the performance upper bound.

**2. Federated Algorithms:**

- **Federated Averaging (FedAvg):** The baseline FL algorithm.
- **Federated Proximal (FedProx):** Addresses heterogeneity by adding a proximal term to the local loss.

**3. Evaluation Metrics:** It's critical to go beyond global accuracy.

- **Global Metrics:** Average performance of the global model on all test sets.
- **Per-Client Metrics:** Performance of the global model on each client's test set.
- **Mean Client Performance:** The average of all per-client metrics. This is a fairer assessment of a global model's utility.
- **Worst-Client Performance:** The minimum performance across all clients. Essential for fairness.
- **Communication Cost:** Number of rounds to convergence.

**Table 12: Experimental Design for FL Evaluation**

| Experiment | Description | Global Test | Per-Client Test | Worst-Client |
| :--- | :--- | :--- | :--- | :--- |
| **Centralized** | Model trained on all data | ✔ (Upper Bound) | - | - |
| **Local-Only** | Independent model per client | - | ✔ | ✔ |
| **FedAvg** | Federated averaging | ✔ | ✔ | ✔ |
| **FedProx** | Federated with proximal term | ✔ | ✔ | ✔ |

---

## 15. Limitations and Threats to Validity

While the EIFD is a powerful tool, its limitations must be carefully considered.

1.  **Synthetic Data:** The data is generated, not sourced from real insurance records. Therefore, the external validity of any findings is limited. Results from this dataset should be interpreted as a proof of concept for algorithms rather than a definitive statement about the real world.
2.  **Data Distribution:** The distributions are generated based on statistical rules. They may not capture the full complexity or the unique edge cases present in real-world data.
3.  **Missing Data Mechanism:** The mechanism for generating missing values is not public. It is likely MCAR or MAR, but this assumption should be confirmed by further investigation.
4.  **Privacy:** The federated architecture is a privacy-enhancing *design*, but the dataset itself is not privacy-preserving. The `parquet` files are unencrypted and can be opened. The privacy guarantee is only realized by the FL *process*. The dataset only provides the architecture for evaluating this process.
5.  **Schema Discrepancy:** The reason the centralized dataset has 47 features while each client has 46 requires investigation. It may be due to the addition of a `client_id` column during centralization, or a bug in the generation process. This is a critical issue for reproducibility.
6.  **Institutional Names:** While the dataset uses the names of real insurance companies for contextual grounding, the data is entirely synthetic and does not represent actual customer information from these institutions. Any analyses or conclusions drawn from this dataset should not be interpreted as reflecting the real-world operations or customer bases of Allianz, AXA, Ageas, or Helvetia.

---

## 16. Reproducibility and Data Access

The entire dataset generation and analysis pipeline is designed for full reproducibility. The EIFD v1.0 can be precisely recreated by executing the following steps:

1.  **Prerequisites:** Python 3.11+ and a compatible environment.
2.  **Clone the Repository:** Obtain the project source code.
3.  **Execute the Generation Script:**
    ```bash
    ./scripts/generator.sh
    ```
    This script will set up the virtual environment, install all necessary dependencies, generate the data, and run the analysis as documented. The exact versions of all key dependencies are printed to the log.

To access the dataset, simply locate the `data/processed/european_insurance_federated/` directory after running the generation script. The data is in Parquet format and can be read with any standard data science library.

---

## 17. Conclusion

The European Insurance Federated Dataset (EIFD) v1.0 is a substantial contribution to the federated learning research community, providing a realistic, heterogeneous, and privacy-relevant benchmark for evaluating state-of-the-art algorithms. The dataset successfully simulates the core challenges of federated learning in a domain—insurance—where data sensitivity and data silos are paramount.

The analysis confirms the dataset's primary utility: its inherent client heterogeneity. Across the four simulated European insurance clients, we observe significant variations in both feature distributions (covariate shift) and target distributions (label shift), creating a non-IID data setting. This non-IID nature makes the dataset an ideal testbed for developing and stress-testing federated optimization algorithms, especially those designed to handle non-convex objectives, statistical heterogeneity, and client drift.

Furthermore, the inclusion of multiple prediction targets—ranging from regression to binary and multi-class classification—provides a comprehensive framework for benchmarking. The dataset is well-structured for reproducibility, offering clear data splits, metadata, and a high-quality centralized baseline. This baseline allows researchers to quantitatively measure the trade-off between the privacy benefits of federated learning and its potential performance degradation due to heterogeneity.

The EIFD v1.0 is more than just a collection of files; it is a robust scientific instrument for investigating fundamental questions in distributed machine learning. By providing this resource to the community, we aim to accelerate progress in federated learning, enabling the development of more robust, fair, and private models for the insurance industry and beyond. Future work will focus on exploring the causes of the schema discrepancy, expanding the dataset to include more clients or more sophisticated synthetic data, and developing and benchmarking novel FL algorithms with this dataset as a primary testbed.

---

## 18. References

1.  McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In *Artificial intelligence and statistics* (pp. 1273-1282). PMLR.
2.  Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., ... & Zhao, S. (2021). Advances and open problems in federated learning. *Foundations and Trends® in Machine Learning*, *14*(1–2), 1-210.
3.  Bühlmann, P., & Gisler, A. (2005). *A course in credibility theory and its applications*. Springer Science & Business Media.
4.  Voigt, P., & Von dem Bussche, A. (2017). The eu general data protection regulation (gdpr). *A Practical Guide, 1st Ed., Cham: Springer International Publishing*, 10(3152676), 10-5555.

---

## 19. Appendices

### Appendix A: Complete Data Dictionary

*(Refer to Table 4 in the main report for the complete data dictionary.)*

### Appendix B: Complete Statistical Tables

*(Refer to Tables 5, 6, and 7 in the main report for detailed statistical tables.)*

### Appendix C: Figure Catalogue

| Figure Number | Title | File Path |
| :--- | :--- | :--- |
| Figure 1 | Client Distribution in Centralized Dataset | `outputs/reports/analysis/centralized/plots/centralized_client_distribution.png` |
| Figure 2 | Feature Distributions - Centralized Dataset | `outputs/reports/analysis/centralized/plots/centralized_distributions.png` |
| Figure 3 | Target Distributions - Centralized Dataset | `outputs/reports/analysis/centralized/plots/centralized_targets.png` |
| Figure 4 | Feature Distributions - Client 1 (Germany) | `outputs/reports/analysis/clients/germany_allianz/plots/distributions_germany_allianz.png` |
| Figure 5 | Correlation Matrix - Client 1 (Germany) | `outputs/reports/analysis/clients/germany_allianz/plots/correlation_germany_allianz.png` |
| Figure 6 | Target Distributions - Client 1 (Germany) | `outputs/reports/analysis/clients/germany_allianz/plots/targets_germany_allianz.png` |
| Figure 7 | Feature Distributions - Client 2 (France) | `outputs/reports/analysis/clients/france_axa/plots/distributions_france_axa.png` |
| Figure 8 | Correlation Matrix - Client 2 (France) | `outputs/reports/analysis/clients/france_axa/plots/correlation_france_axa.png` |
| Figure 9 | Target Distributions - Client 2 (France) | `outputs/reports/analysis/clients/france_axa/plots/targets_france_axa.png` |
| Figure 10 | Feature Distributions - Client 3 (Belgium) | `outputs/reports/analysis/clients/belgium_ageas/plots/distributions_belgium_ageas.png` |
| Figure 11 | Correlation Matrix - Client 3 (Belgium) | `outputs/reports/analysis/clients/belgium_ageas/plots/correlation_belgium_ageas.png` |
| Figure 12 | Target Distributions - Client 3 (Belgium) | `outputs/reports/analysis/clients/belgium_ageas/plots/targets_belgium_ageas.png` |
| Figure 13 | Feature Distributions - Client 4 (Switzerland) | `outputs/reports/analysis/clients/switzerland_helvetia/plots/distributions_switzerland_helvetia.png` |
| Figure 14 | Correlation Matrix - Client 4 (Switzerland) | `outputs/reports/analysis/clients/switzerland_helvetia/plots/correlation_switzerland_helvetia.png` |
| Figure 15 | Target Distributions - Client 4 (Switzerland) | `outputs/reports/analysis/clients/switzerland_helvetia/plots/targets_switzerland_helvetia.png` |
| Figure 16 | Missing Values - Client 2 (France) | `outputs/reports/analysis/clients/france_axa/plots/missing_values_france_axa.png` |
| Figure 17 | Missing Values - Client 1 (Germany) | `outputs/reports/analysis/clients/germany_allianz/plots/missing_values_germany_allianz.png` |
| Figure 18 | Missing Values - Client 4 (Switzerland) | `outputs/reports/analysis/clients/switzerland_helvetia/plots/missing_values_switzerland_helvetia.png` |