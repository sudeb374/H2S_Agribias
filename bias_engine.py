import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import fairlearn.metrics as fl_metrics
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import shap
import json

def run_bias_pipeline(data_path="synthetic_pmfby_data.csv"):
    df = pd.read_csv(data_path)
    
    # 1. Preprocessing
    # Features (X) & Target (y)
    categorical_cols = ['State', 'Crop', 'Irrigation_Type', 'Farmer_Scale']
    
    # Create copies for Fairlearn
    A_irrigation = df['Irrigation_Type']
    
    # Predict Insurance_Approval
    # We will encode categoricals
    X = pd.get_dummies(df[['State', 'Crop', 'Irrigation_Type', 'Farmer_Scale', 'Crop_Damage_Pct', 'Advisory_Score']], 
                       columns=categorical_cols, drop_first=True)
    y = df['Insurance_Approval']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A_irrigation, test_size=0.3, random_state=42
    )

    # 2. Train baseline XGBoost
    xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_base.fit(X_train, y_train)
    y_pred_base = xgb_base.predict(X_test)
    
    base_acc = accuracy_score(y_test, y_pred_base)
    
    # 3. Compute Fairness Metrics (Fairlearn)
    # Using Demographic Parity Difference and Disparate Impact Ratio on Irrigation Type
    dp_diff_base = fl_metrics.demographic_parity_difference(y_test, y_pred_base, sensitive_features=A_test)
    dp_ratio_base = fl_metrics.demographic_parity_ratio(y_test, y_pred_base, sensitive_features=A_test)
    
    # Equalized odds difference
    eo_diff_base = fl_metrics.equalized_odds_difference(y_test, y_pred_base, sensitive_features=A_test)

    # Compute custom Equity Score (0-100) based on these metrics
    # A perfectly equitable model has DP ratio close to 1, DP diff and EO diff close to 0
    base_equity_score = max(0, min(100, 100 * (1 - ((dp_diff_base + eo_diff_base) / 2))))

    # 4. SHAP Feature Importance
    explainer = shap.Explainer(xgb_base)
    shap_values = explainer(X_test)
    
    # Get mean absolute SHAP values per feature
    feature_names = X.columns
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    
    shap_dict = {feat: float(val) for feat, val in zip(feature_names, mean_shap)}
    # Sort SHAP dict
    shap_dict = dict(sorted(shap_dict.items(), key=lambda item: item[1], reverse=True))

    # Flag top features (Proxy Variables) if they are categorical and have high shap contribution
    total_shap = sum(shap_dict.values())
    proxies_flagged = []
    for k, v in shap_dict.items():
        if (v / total_shap) > 0.15 and ('Irrigation' in k or 'Farmer' in k or 'State' in k):
            proxies_flagged.append(k)
            
    # 5. MITIGATION: Fairlearn Reductions
    mitigator = ExponentiatedGradient(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        constraints=DemographicParity()
    )
    # Train mitigated model
    mitigator.fit(X_train, y_train, sensitive_features=A_train)
    y_pred_mitigated = mitigator.predict(X_test)
    
    mit_acc = accuracy_score(y_test, y_pred_mitigated)
    dp_diff_mit = fl_metrics.demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=A_test)
    dp_ratio_mit = fl_metrics.demographic_parity_ratio(y_test, y_pred_mitigated, sensitive_features=A_test)
    eo_diff_mit = fl_metrics.equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=A_test)
    
    mit_equity_score = max(0, min(100, 100 * (1 - ((dp_diff_mit + eo_diff_mit) / 2))))

    # Result assembly
    results = {
        "Baseline": {
            "Accuracy": round(base_acc, 3),
            "Demographic_Parity_Diff": round(dp_diff_base, 3),
            "Disparate_Impact_Ratio": round(dp_ratio_base, 3),
            "Equalized_Odds_Diff": round(eo_diff_base, 3),
            "Equity_Score": round(base_equity_score, 1)
        },
        "Mitigated": {
            "Accuracy": round(mit_acc, 3),
            "Demographic_Parity_Diff": round(dp_diff_mit, 3),
            "Disparate_Impact_Ratio": round(dp_ratio_mit, 3),
            "Equalized_Odds_Diff": round(eo_diff_mit, 3),
            "Equity_Score": round(mit_equity_score, 1)
        },
        "SHAP_Analysis": {
            "Top_Features": list(shap_dict.keys())[:5],
            "Flagged_Proxies": proxies_flagged
        }
    }
    
    # Save to JSON for agents to easily ingest
    with open('bias_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    res = run_bias_pipeline()
    print("Bias pipeline completed. Equity improvement:", 
          f"{res['Baseline']['Equity_Score']} -> {res['Mitigated']['Equity_Score']}")
