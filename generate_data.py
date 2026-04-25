import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

NUM_ROWS = np.random.randint(1000, 2000)

# Generating base features
all_states = ['Punjab', 'Haryana', 'Odisha', 'Jharkhand', 'Chhattisgarh', 
              'Maharashtra', 'Uttar Pradesh', 'Madhya Pradesh', 'Rajasthan', 
              'Gujarat', 'Karnataka', 'Andhra Pradesh', 'Tamil Nadu', 
              'Bihar', 'West Bengal', 'Assam']

# Slight over-representation of low-data states to see if error rate differs
# There are 16 states. Let's give low-data states slightly higher probability
low_data_states = ['Odisha', 'Jharkhand', 'Chhattisgarh', 'Bihar', 'Assam']
high_data_states = [s for s in all_states if s not in low_data_states]

probs = [0.10 if s in low_data_states else 0.50 / len(high_data_states) for s in all_states]

states = np.random.choice(
    all_states, 
    size=NUM_ROWS, 
    p=probs
)

all_crops = ['Cotton', 'Wheat', 'Rice', 'Soybean', 'Millet', 'Sugarcane', 
             'Maize', 'Barley', 'Jute', 'Groundnut', 'Mustard', 'Gram', 
             'Tur', 'Moong']
crops = np.random.choice(all_crops, size=NUM_ROWS)

irrigation = np.random.choice(['Rain-fed', 'Irrigated'], size=NUM_ROWS, p=[0.6, 0.4])

farmer_scale = np.random.choice(['Small/Marginal', 'Large'], size=NUM_ROWS, p=[0.75, 0.25])

# Crop damage uniformly distributed
crop_damage_pct = np.random.uniform(0, 100, size=NUM_ROWS)

insurance_approvals = []
advisory_scores = []

for i in range(NUM_ROWS):
    # Base probability of approval correlates highly with crop damage
    approval_prob = (crop_damage_pct[i] / 100) * 0.8  # Max 80% base chance
    
    # Base advisory score based on random normally distributed around 60
    base_advisory = np.random.normal(60, 15)
    
    # INJECT BIAS 1: Rain-fed vs Irrigated
    # Rain-fed farmers randomly get penalised in approvals
    if irrigation[i] == 'Rain-fed':
        approval_prob -= 0.15
        base_advisory -= 5
    else:
        approval_prob += 0.10
        base_advisory += 10
        
    # INJECT BIAS 2: Farmer Scale
    # Small/Marginal farmers have higher rejection
    if farmer_scale[i] == 'Small/Marginal':
        approval_prob -= 0.10
        base_advisory -= 10
    else:
        approval_prob += 0.15
        base_advisory += 15
        
    # INJECT BIAS 3: State-based data poverty
    # Low data states get worse recommendations (lower advisory score), and slight penalty in insurance
    if states[i] in low_data_states:
        base_advisory -= 15
        approval_prob -= 0.05
    else:
        base_advisory += 10
        approval_prob += 0.10
        
    # Cap probabilities and scores
    approval_prob = max(0.01, min(0.99, approval_prob))
    base_advisory = max(0.0, min(100.0, base_advisory))
    
    # Bernoulli trial for approval
    approval = np.random.binomial(1, approval_prob)
    
    insurance_approvals.append(approval)
    advisory_scores.append(round(base_advisory, 2))

# Construct dataframe
df = pd.DataFrame({
    'Farmer_ID': [f'F_{i+1000}' for i in range(NUM_ROWS)],
    'State': states,
    'Crop': crops,
    'Irrigation_Type': irrigation,
    'Farmer_Scale': farmer_scale,
    'Crop_Damage_Pct': np.round(crop_damage_pct, 2),
    'Advisory_Score': advisory_scores,
    'Insurance_Approval': insurance_approvals
})

df.to_csv('synthetic_pmfby_data.csv', index=False)
print(f"Saved synthetic_pmfby_data.csv with {NUM_ROWS} rows.")
