# Project Pitch: Auditing AI Bias in Agricultural Yield and Insurance Systems

## 1. The Challenge
India's agricultural sector is increasingly reliant on data-driven models for issuing advisories and processing Pradhan Mantri Fasal Bima Yojana (PMFBY) insurance claims. However, algorithmic systems can inadvertently encode historical inequities, disproportionately impacting vulnerable farming communities.

## 2. Identified Disparities
Our analysis focuses on detecting and mitigating three core bias problem sets within current agricultural AI outputs:

1. **Irrigation Disparities (The Counterfactual Gap):** 
   Rain-fed farmers and irrigated farmers experiencing identical percentage-based crop damage often receive diverging advisory scores and insurance payout approvals. Rain-fed farmers consistently show higher rejection rates, signaling that the model leverages irrigation status as a negative proxy.
2. **Farmer Scale Inequities (The Demographic Parity Gap):**
   Small and marginal farmers experience higher insurance claim rejection rates and lower automated advisory scores compared to large-scale farmers, despite similar fundamental distress markers.
3. **Geographic Data Poverty:**
   Farmers in "low-data" regions like Odisha, Jharkhand, and Chhattisgarh frequently receive lower-quality and systematically biased recommendations compared to data-rich states like Punjab and Haryana, primarily due to sampling deficits in the training data leading to increased false-negative rates for the marginalized regions.

## 3. Our Solution
We propose an automated, multi-agent equity auditing pipeline utilizing the Agentic framework (AutoGen AG2) combined with Fairlearn. The system ingests crop damage and payout data, automatically surfaces discriminatory proxy variables using SHAP, formally calculates the disparate impact ratio, and deploys mitigation algorithms to retrain a corrected model.

## 4. Impact
By mandating these automated, verifiable bias audits, regulatory bodies (such as the Ministry of Agriculture) can guarantee that ML-assisted PMFBY systems remain fair, accountable, and transparent across the diverse demographics of Indian agriculture.
