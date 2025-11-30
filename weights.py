import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = '/Users/debasmibasu/Documents/maths - lpp/Attendance_Optimization_bootstrapped_60.csv'
df = pd.read_csv(file_path)

# Define the 8 key factors based on your survey:
# 1. Perceived Value/Learning
# 2. Liking & Engagement  
# 3. Study Time Efficiency
# 4. Attendance Risk
# 5. Time Block Preference (Morning/Mid-day/Afternoon/Late-afternoon)
# 6. Holiday Skip Likelihood
# 7. Travel Time (from initial questions)
# 8. Time Commitments (from initial questions)

# Example: Creating aggregate scores per student per professor
# You'll need to adapt this to your actual data structure

def calculate_correlation_matrix(df):
    """
    Calculate correlation matrix for the 8 attendance factors
    
    Parameters:
    df: DataFrame with all survey responses
    
    Returns:
    correlation_matrix: DataFrame with correlations between factors
    """
    
    # Calculate average scores across all professors for each factor
    factor_scores = pd.DataFrame()
    
    # Factor 1: Perceived Value/Learning
    perceived_value_cols = [col for col in df.columns if 'Perceived value' in col]
    if perceived_value_cols:
        factor_scores['Perceived_Value'] = df[perceived_value_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    
    # Factor 2: Liking & Engagement
    engagement_cols = [col for col in df.columns if 'Liking & Engagement' in col]
    if engagement_cols:
        factor_scores['Liking_Engagement'] = df[engagement_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    
    # Factor 3: Study Time Efficiency
    efficiency_cols = [col for col in df.columns if 'Study Time Efficiency' in col]
    if efficiency_cols:
        factor_scores['Study_Efficiency'] = df[efficiency_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    
    # Factor 4: Attendance Risk
    risk_cols = [col for col in df.columns if 'Attendance Risk' in col]
    if risk_cols:
        factor_scores['Attendance_Risk'] = df[risk_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    
    # Factor 5: Time Block Preference (convert to numeric: higher = prefer later times)
    time_block_cols = [col for col in df.columns if 'Time Block' in col]
    if time_block_cols:
        factor_scores['Time_Block_Pref'] = df[time_block_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    
    # Factor 6: Holiday Skip Likelihood
    holiday_cols = [col for col in df.columns if 'Holiday Skip' in col]
    if holiday_cols:
        factor_scores['Holiday_Skip'] = df[holiday_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    
    # Factor 7: Travel Time (convert categorical to numeric)
    travel_col = 'What is your typical ONE-WAY travel time to college on an average day?'
    if travel_col in df.columns:
        travel_mapping = {
            'Under 15 minutes': 1,
            '15 - 30 minutes': 2,
            '30 - 60 minutes': 3,
            '60 to 90 minutes': 4,
            'over 90 minutes': 5
        }
        factor_scores['Travel_Time'] = df[travel_col].map(travel_mapping)
    
    # Factor 8: Time Commitments (convert categorical to numeric using the formula from image)
    # TT = 0 (only college), 0.5 (college + society/clubs), 1 (college + part-time), 1.5 (college + family issues)
    commitment_col = None
    for col in df.columns:
        if 'major time commitments' in col.lower():
            commitment_col = col
            break
    
    if commitment_col is not None:
        print(f"\nFound Time Commitments column: {commitment_col}")
        print(f"Unique values in column:")
        print(df[commitment_col].value_counts())
        
        # Create mapping with partial string matching
        def map_commitment(value):
            if pd.isna(value):
                return np.nan
            value_str = str(value).lower()
            if 'only major commitment' in value_str or 'no,' in value_str:
                return 0.0
            elif 'society' in value_str or 'club' in value_str or 'sports' in value_str:
                return 0.5
            elif 'part-time' in value_str or 'internship' in value_str or 'job' in value_str:
                return 1.0
            elif 'family' in value_str or 'personal' in value_str:
                return 1.5
            else:
                return np.nan
        
        factor_scores['Time_Commitments'] = df[commitment_col].apply(map_commitment)
        print(f"Time Commitments mapped: {factor_scores['Time_Commitments'].notna().sum()} values")
        print(f"Distribution: \n{factor_scores['Time_Commitments'].value_counts().sort_index()}")
    else:
        print(f"WARNING: Could not find column for Time Commitments")
        print(f"Available columns: {df.columns.tolist()[:10]}")
    
    # Remove any rows with all NaN values
    factor_scores = factor_scores.dropna(how='all')
    
    # Calculate correlation matrix
    correlation_matrix = factor_scores.corr()
    
    return correlation_matrix, factor_scores


def normalize_correlation_matrix(correlation_matrix):
    """
    Normalize correlation matrix by dividing each entry by the sum of its column
    Then calculate weights as row averages
    
    Parameters:
    correlation_matrix: DataFrame with correlation values
    
    Returns:
    normalized_matrix: Normalized correlation matrix
    weights: Series with factor weights (row averages)
    """
    
    # Take absolute values for normalization (since correlations can be negative)
    abs_corr_matrix = correlation_matrix.abs()
    
    # Normalize: divide each entry by its column sum
    column_sums = abs_corr_matrix.sum(axis=0)
    normalized_matrix = abs_corr_matrix / column_sums
    
    # Calculate weights as row averages of normalized matrix
    weights = normalized_matrix.mean(axis=1)
    
    # Normalize weights to sum to 1 (if needed)
    weights_normalized = weights / weights.sum()
    
    return normalized_matrix, weights, weights_normalized


def plot_normalized_matrix(normalized_matrix, save_path='normalized_correlation_matrix.png'):
    """
    Create a heatmap visualization of the normalized correlation matrix
    
    Parameters:
    normalized_matrix: DataFrame with normalized correlation values
    save_path: Path to save the figure
    """
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(normalized_matrix, 
                annot=True,  # Show values
                fmt='.4f',   # Format to 4 decimal places
                cmap='YlOrRd',  # Yellow-Orange-Red colormap
                square=True,  # Make cells square
                linewidths=1,  # Add gridlines
                cbar_kws={"shrink": 0.8})
    
    plt.title('Normalized Correlation Matrix\n(Each entry divided by column sum)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Factors', fontsize=12, fontweight='bold')
    plt.ylabel('Factors', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Normalized correlation matrix saved to {save_path}")
    plt.show()


def plot_weights_bar_chart(weights_normalized, save_path='factor_weights.png'):
    """
    Create a bar chart of factor weights
    
    Parameters:
    weights_normalized: Series with normalized weights
    save_path: Path to save the figure
    """
    
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    bars = plt.bar(range(len(weights_normalized)), 
                   weights_normalized.values,
                   color='steelblue',
                   edgecolor='navy',
                   linewidth=1.5)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Factors', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Weight', fontsize=12, fontweight='bold')
    plt.title('Factor Weights (Row Averages of Normalized Matrix)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(weights_normalized)), 
               weights_normalized.index,
               rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Factor weights chart saved to {save_path}")
    plt.show()


# MAIN EXECUTION
print("Loading data...")
df = pd.read_csv(file_path)
print(f"Data loaded successfully! Shape: {df.shape}")
print(f"Number of responses: {len(df)}")

# Calculate correlation matrix
print("\nCalculating correlation matrix...")
correlation_matrix, factor_scores = calculate_correlation_matrix(df)

# Debug: Check which factors were successfully calculated
print("\n" + "="*60)
print("FACTORS INCLUDED IN ANALYSIS:")
print("="*60)
for col in factor_scores.columns:
    non_null_count = factor_scores[col].notna().sum()
    print(f"{col}: {non_null_count} valid responses")
print(f"\nTotal factors: {len(factor_scores.columns)}")
print(f"Expected: 8 factors")

# Print original correlation matrix
print("\n" + "="*60)
print("ORIGINAL CORRELATION MATRIX:")
print("="*60)
print(correlation_matrix.round(3))

# Normalize correlation matrix and calculate weights
print("\nNormalizing correlation matrix and calculating weights...")
normalized_matrix, weights, weights_normalized = normalize_correlation_matrix(correlation_matrix)

# Print normalized matrix
print("\n" + "="*60)
print("NORMALIZED CORRELATION MATRIX:")
print("(Each entry divided by column sum)")
print("="*60)
print(normalized_matrix.round(4))

# Print column sums to verify normalization
print("\n" + "="*60)
print("COLUMN SUMS (should all be 1.0):")
print("="*60)
print(normalized_matrix.sum(axis=0).round(4))

# Print weights
print("\n" + "="*60)
print("FACTOR WEIGHTS (Row Averages):")
print("="*60)
weights_df = pd.DataFrame({
    'Factor': weights.index,
    'Weight (Raw)': weights.values,
    'Weight (Normalized)': weights_normalized.values,
    'Weight (%)': (weights_normalized.values * 100)
})
print(weights_df.to_string(index=False))

# Print ranked weights
print("\n" + "="*60)
print("FACTORS RANKED BY IMPORTANCE:")
print("="*60)
weights_ranked = weights_df.sort_values('Weight (Normalized)', ascending=False)
for idx, row in weights_ranked.iterrows():
    print(f"{row['Factor']:.<30} {row['Weight (%)']:.2f}%")

# Plot original correlation matrix
print("\nGenerating visualizations...")
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)
plt.title('Original Correlation Matrix: Attendance Optimization Factors', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Factors', fontsize=12, fontweight='bold')
plt.ylabel('Factors', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Original correlation matrix saved to 'correlation_matrix.png'")
plt.show()

# Plot normalized matrix
plot_normalized_matrix(normalized_matrix)

# Plot weights
plot_weights_bar_chart(weights_normalized)

# Summary statistics
print("\n" + "="*60)
print("FACTOR SCORE STATISTICS:")
print("="*60)
print(factor_scores.describe().round(2))

# Identify strongest correlations in original matrix
print("\n" + "="*60)
print("STRONGEST CORRELATIONS (Original Matrix):")
print("="*60)
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_pairs.append({
            'Factor 1': correlation_matrix.columns[i],
            'Factor 2': correlation_matrix.columns[j],
            'Correlation': correlation_matrix.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs)
corr_df = corr_df.sort_values('Correlation', ascending=False)
print("\nTop 5 Positive Correlations:")
print(corr_df.head(5).to_string(index=False))
print("\nTop 5 Negative Correlations:")
print(corr_df.tail(5).to_string(index=False))

print("\n" + "="*60)
print("Analysis complete!")
print("Generated files:")
print("  - correlation_matrix.png (original)")
print("  - normalized_correlation_matrix.png")
print("  - factor_weights.png")
print("="*60)