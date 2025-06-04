import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Dummy dataset generation
n_rows = 1000
df = pd.DataFrame({
    'Age': np.random.randint(18, 70, size=n_rows),
    'Income': np.random.normal(loc=60000, scale=15000, size=n_rows).round(2),
    'Gender': np.random.choice(['Male', 'Female', 'Other'], size=n_rows),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n_rows),
    'PurchaseAmount': np.random.exponential(scale=500, size=n_rows).round(2),
    'Membership': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=n_rows),
    'Churn': np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2])
})

# Save it to CSV
df.to_csv("dummy_data.csv", index=False)
print("Dummy dataset saved to dummy_data.csv ✅")
