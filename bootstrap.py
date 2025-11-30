import pandas as pd

# Load your dataset
df = pd.read_csv("Attendance Optimization.csv")   # or pd.read_excel(...)

# Desired size
target_size = 60

# Bootstrap sample (sampling with replacement)
bootstrapped_df = df.sample(n=target_size, replace=True, random_state=42)

# Save the new bootstrapped dataset
bootstrapped_df.to_csv("Attendance_Optimization_bootstrapped_60.csv", index=False)

print("Bootstrapped dataset created with 60 rows.")
