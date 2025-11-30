import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = '/Users/debasmibasu/Documents/maths - lpp/Attendance_Optimization_bootstrapped_60.csv'
df = pd.read_csv(file_path)

print("="*80)
print("PROFESSOR-SPECIFIC FACTOR ANALYSIS")
print("="*80)
print(f"Data loaded successfully! Shape: {df.shape}")
print(f"Number of responses: {len(df)}\n")

# Define the 16 professors
professors = [
    'Prof. B. Biswal',
    'Prof. Shobha Bagai',
    'Prof. Pankaj Tyagi',
    'Prof. Swati Arora',
    'Prof. Mahima Kaushik',
    'Prof. Nirmal Yadav',
    'Prof. Sonam Tanwar',
    'Prof. Asani Bhaduri',
    'Prof. Harendra Pal Singh',
    'Prof. Sachin Kumar',
    'Prof. J.S. Purohit',
    'Prof. Dorje Dawa',
    'Prof. Shobha Rai',
    'Prof. Anjani Verma',
    'Prof. Manish Kumar',
    'Sanjeewani Sehgal'
]

# Define the 6 professor-related factors
factor_types = {
    'Perceived_Value': 'Perceived value/ learning:',
    'Liking_Engagement': 'Liking & Engagement:',
    'Study_Efficiency': 'Study Time Efficiency:',
    'Attendance_Risk': 'Attendance Risk:',
    'Time_Block_Morning': 'Time Block: Morning\n9 AM - 11 AM',
    'Time_Block_Midday': 'Time Block: Mid-Day\n(Pre- Lunch)\n11 AM - 1 PM',
    'Time_Block_Afternoon': 'Time Block: Mid-Afternoon\n(Post- Lunch)\n2 PM - 4 PM',
    'Time_Block_Late': 'Time Block: Late-Afternoon\n4 PM - 6 PM',
    'Holiday_Skip': 'Holiday Skip Likelihood'
}


def calculate_professor_means():
    """
    Calculate mean scores for each of the 6 factors for all 16 professors
    
    Returns:
    results_df: DataFrame with professors as rows and factors as columns
    """
    
    results = []
    
    for professor in professors:
        prof_data = {}
        prof_data['Professor'] = professor
        
        # Factor 1: Perceived Value/Learning
        perceived_cols = [col for col in df.columns if 'Perceived value' in col and professor in col]
        if perceived_cols:
            prof_data['Perceived_Value'] = df[perceived_cols[0]].apply(pd.to_numeric, errors='coerce').mean()
        else:
            prof_data['Perceived_Value'] = np.nan
        
        # Factor 2: Liking & Engagement
        engagement_cols = [col for col in df.columns if 'Liking & Engagement' in col and professor in col]
        if engagement_cols:
            prof_data['Liking_Engagement'] = df[engagement_cols[0]].apply(pd.to_numeric, errors='coerce').mean()
        else:
            prof_data['Liking_Engagement'] = np.nan
        
        # Factor 3: Study Time Efficiency
        efficiency_cols = [col for col in df.columns if 'Study Time Efficiency' in col and professor in col]
        if efficiency_cols:
            prof_data['Study_Efficiency'] = df[efficiency_cols[0]].apply(pd.to_numeric, errors='coerce').mean()
        else:
            prof_data['Study_Efficiency'] = np.nan
        
        # Factor 4: Attendance Risk
        risk_cols = [col for col in df.columns if 'Attendance Risk' in col and professor in col]
        if risk_cols:
            prof_data['Attendance_Risk'] = df[risk_cols[0]].apply(pd.to_numeric, errors='coerce').mean()
        else:
            prof_data['Attendance_Risk'] = np.nan
        
        # Factor 5: Time Block Preference (average across all time blocks)
        time_cols = [col for col in df.columns if 'Time Block' in col and professor in col]
        if time_cols:
            time_values = []
            for col in time_cols:
                vals = df[col].apply(pd.to_numeric, errors='coerce')
                time_values.extend(vals.dropna().tolist())
            prof_data['Time_Block_Preference'] = np.mean(time_values) if time_values else np.nan
        else:
            prof_data['Time_Block_Preference'] = np.nan
        
        # Factor 6: Holiday Skip Likelihood
        holiday_cols = [col for col in df.columns if 'Holiday Skip' in col and professor in col]
        if holiday_cols:
            prof_data['Holiday_Skip_Likelihood'] = df[holiday_cols[0]].apply(pd.to_numeric, errors='coerce').mean()
        else:
            prof_data['Holiday_Skip_Likelihood'] = np.nan
        
        results.append(prof_data)
    
    results_df = pd.DataFrame(results)
    return results_df


def plot_professor_comparison(results_df):
    """
    Create visualizations comparing professors across all factors
    """
    
    # Prepare data for plotting (exclude Professor column)
    plot_data = results_df.set_index('Professor')
    
    # 1. Heatmap of all professors and factors
    plt.figure(figsize=(14, 10))
    sns.heatmap(plot_data, 
                annot=True, 
                fmt='.2f', 
                cmap='RdYlGn',
                center=5,
                vmin=1, 
                vmax=10,
                linewidths=0.5,
                cbar_kws={'label': 'Score (1-10)'})
    plt.title('Professor Performance Across 6 Key Factors\n(Higher scores = Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Factors', fontsize=12, fontweight='bold')
    plt.ylabel('Professors', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('professor_factor_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved to 'professor_factor_heatmap.png'")
    plt.show()
    
    # 2. Bar chart for each factor
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    factors = plot_data.columns
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for idx, (factor, color) in enumerate(zip(factors, colors)):
        ax = axes[idx]
        data_sorted = plot_data[factor].sort_values(ascending=False)
        
        bars = ax.barh(range(len(data_sorted)), data_sorted.values, color=color, edgecolor='black')
        ax.set_yticks(range(len(data_sorted)))
        ax.set_yticklabels(data_sorted.index, fontsize=9)
        ax.set_xlabel('Mean Score', fontsize=10, fontweight='bold')
        ax.set_title(factor.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 10)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, data_sorted.values)):
            if not np.isnan(val):
                ax.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('professor_factor_bars.png', dpi=300, bbox_inches='tight')
    print("Bar charts saved to 'professor_factor_bars.png'")
    plt.show()
    
    # 3. Radar chart for top 5 professors
    top_5_profs = results_df.set_index('Professor').mean(axis=1).nlargest(5).index
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    categories = list(plot_data.columns)
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot data
    for prof in top_5_profs:
        values = plot_data.loc[prof].values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=prof)
        ax.fill(angles, values, alpha=0.15)
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([cat.replace('_', '\n') for cat in categories], fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_title('Top 5 Professors - Radar Chart Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('professor_radar_chart.png', dpi=300, bbox_inches='tight')
    print("Radar chart saved to 'professor_radar_chart.png'")
    plt.show()


def generate_summary_statistics(results_df):
    """
    Generate detailed summary statistics for professor performance
    """
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY FACTOR")
    print("="*80)
    
    plot_data = results_df.set_index('Professor')
    
    for factor in plot_data.columns:
        print(f"\n{factor.replace('_', ' ').upper()}:")
        print("-" * 80)
        factor_data = plot_data[factor].dropna()
        
        print(f"Mean:   {factor_data.mean():.3f}")
        print(f"Median: {factor_data.median():.3f}")
        print(f"Std:    {factor_data.std():.3f}")
        print(f"Min:    {factor_data.min():.3f} ({factor_data.idxmin()})")
        print(f"Max:    {factor_data.max():.3f} ({factor_data.idxmax()})")
    
    # Overall professor rankings
    print("\n" + "="*80)
    print("OVERALL PROFESSOR RANKINGS (Average across all 6 factors)")
    print("="*80)
    
    results_df['Overall_Average'] = results_df.iloc[:, 1:].mean(axis=1)
    ranked = results_df[['Professor', 'Overall_Average']].sort_values('Overall_Average', ascending=False)
    
    for rank, row in enumerate(ranked.itertuples(), 1):
        print(f"{rank:2d}. {row.Professor:<30} {row.Overall_Average:.3f}")
    
    return results_df


# MAIN EXECUTION
print("Calculating mean scores for each professor across 6 factors...")
results_df = calculate_professor_means()

# Display the results table
print("\n" + "="*80)
print("PROFESSOR FACTOR MEANS")
print("="*80)
print(results_df.to_string(index=False, float_format='%.3f'))

# Export to CSV
output_file = 'professor_factor_means.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults exported to '{output_file}'")

# Generate visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)
plot_professor_comparison(results_df)

# Generate summary statistics and rankings
results_df = generate_summary_statistics(results_df)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("Generated files:")
print("  - professor_factor_means.csv")
print("  - professor_factor_heatmap.png")
print("  - professor_factor_bars.png")
print("  - professor_radar_chart.png")
print("="*80)